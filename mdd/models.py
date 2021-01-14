import math

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import models as models

from . import loss
from .utils.utils import GradientReverseLayer, calc_coeff, entropy, init_weights


class MDDLitModel(pl.LightningModule):
    def __init__(
        self,
        feature_ext="resnet50",
        use_bottleneck=True,
        bottleneck_dim=256,
        new_classifier=True,
        num_class=1000,
        random_proj=False,
        random_proj_dim=1024,
        lr=0.001,
        momentum=0.9,
        left_weight=1,
        right_weight=1,
        cls_weight=1,
        mdd_weight=0.01,
        entropic_weight=0,
        loss_trade_off=1,
        scheduler_lr=0.001,
        scheduler_gamma=0.001,
        scheduler_power=0.75,
        scheduler_weight_decay=0.0005,
        max_iter=10000,
        test_10crop=True,
    ):
        super(MDDLitModel, self).__init__()
        self.save_hyperparameters()
        self.num_class = num_class
        self.feature_ext = ResNetFc(
            resnet_name=feature_ext,
            use_bottleneck=use_bottleneck,
            bottleneck_dim=bottleneck_dim,
            new_classifier=new_classifier,
            num_class=num_class,
        )
        if random_proj:
            self.random_layer = RandomLayer(
                self.feature_ext.output_num(),
                num_class,
                random_proj_dim,
            )
            self.adv = AdversarialNetwork(
                random_proj_dim,
                1024,
                random_layer=self.random_layer,
                max_iter=max_iter,
            )
        else:
            self.random_layer = None
            self.adv = AdversarialNetwork(
                self.feature_ext.output_num() * num_class,
                1024,
                random_layer=self.random_layer,
                max_iter=max_iter,
            )

        self.left_weight = left_weight
        self.right_weight = right_weight
        self.cls_weight = cls_weight
        self.mdd_weight = mdd_weight
        self.entropic_weight = entropic_weight
        self.loss_trade_off = loss_trade_off
        self.scheduler_lr = scheduler_lr
        self.scheduler_gamma = scheduler_gamma
        self.scheduler_power = scheduler_power
        self.scheduler_weight_decay = scheduler_weight_decay
        self.max_iter = max_iter
        self.test_10crop = test_10crop

        self.lr = lr
        self.momentum = momentum

        self.accuracy = pl.metrics.Accuracy()

    def forward(self, source, target):
        feat_source, out_source = self.feature_ext(source)
        feat_target, out_target = self.feature_ext(target)
        feat = torch.cat((feat_source, feat_target), dim=0)
        out = torch.cat((out_source, out_target), dim=0)
        softmax_out = F.softmax(out, dim=1)
        labels_target = torch.max(F.softmax(out_target, dim=1), 1)[1]
        return out_source, labels_target, feat, softmax_out

    def training_step(self, batch, batch_idx):
        inputs_source, labels_source = batch[0]
        inputs_target, _ = batch[1]
        out_source, labels_target, feat, softmax_out = self(
            inputs_source, inputs_target
        )
        labels = torch.cat((labels_source, labels_target))
        h = entropy(softmax_out)

        transfer_loss = 0
        if self.loss_trade_off > 0:
            transfer_loss = self.adv.cdan(
                feat,
                softmax_out,
                h,
                calc_coeff(batch_idx, max_iter=self.max_iter),
            )

        classifier_loss = 0
        if self.cls_weight > 0:
            classifier_loss = F.cross_entropy(out_source, labels_source)

        mdd_loss = 0
        if self.mdd_weight > 0:
            mdd_loss = loss.mdd_loss(
                features=feat,
                labels=labels,
                left_weight=self.left_weight,
                right_weight=self.right_weight,
            )

        max_entropy_loss = 0
        if self.entropic_weight > 0:
            max_entropy_loss = loss.entropic_loss(feat)

        total_loss = (
            self.loss_trade_off * transfer_loss
            + self.cls_weight * classifier_loss
            + self.mdd_weight * mdd_loss
            + self.entropic_weight * max_entropy_loss
        )

        self.log(
            "train_loss",
            total_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )
        self.log(
            "cls_loss",
            classifier_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "transfer_loss",
            transfer_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "mdd_loss",
            mdd_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "entropy_loss",
            max_entropy_loss,
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "train_acc",
            self.accuracy(out_source.argmax(1), labels_source),
            on_step=True,
            on_epoch=True,
            logger=True,
        )
        return total_loss

    def test_step(self, batch, batch_idx):
        inputs, labels = batch
        if self.test_10crop:
            bs, ncrops, c, h, w = inputs.size()
            _, outputs = self.feature_ext(inputs.view(-1, c, h, w))
            labels = (
                labels.view(-1, 1)
                .expand(-1, ncrops)
                .contiguous()
                .view(ncrops * bs)
            )
        else:
            _, outputs = self.feature_ext(inputs)
        self.log(
            "test_acc",
            self.accuracy(outputs.argmax(1), labels),
            on_step=True,
            on_epoch=True,
            logger=True,
            prog_bar=True,
        )

    def optimizer_step(
        self,
        current_epoch,
        batch_nb,
        optimizer,
        optimizer_idx,
        closure,
        on_tpu=False,
        using_native_amp=False,
        using_lbfgs=False,
    ) -> None:
        lr = self.scheduler_lr * (1 + self.scheduler_gamma * batch_nb) ** (
            -self.scheduler_power
        )
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr * param_group["lr_mult"]
            param_group["weight_decay"] = (
                self.scheduler_weight_decay * param_group["decay_mult"]
            )
        optimizer.zero_grad(set_to_none=True)
        optimizer.step(closure=closure)

    def configure_optimizers(self):
        params_to_update = (
            self.feature_ext.get_parameters() + self.adv.get_parameters()
        )
        optimizer = optim.SGD(
            params_to_update,
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=0.0005,
            nesterov=True,
        )
        return optimizer

    @staticmethod
    def add_model_specific_args(parser):
        # parser = ArgumentParser(parents=[parent_parser], add_help=False)
        model_args = parser.add_argument_group("model arguments")
        model_args.add_argument(
            "--feature_ext",
            type=str,
            default="resnet50",
            choices=[
                "resnet18",
                "resnet34",
                "resnet50",
                "resnet101",
                "resnet152",
            ],
            help="feature extractor type",
        )
        model_args.add_argument(
            "--use_bottleneck",
            type=bool,
            default=True,
            help="whether to use bottleneck in the classifier",
        )
        model_args.add_argument(
            "--bottleneck_dim",
            type=int,
            default=256,
            help="whether to use bottleneck in the classifier",
        )
        model_args.add_argument(
            "--new_classifier",
            type=bool,
            default=True,
            help="whether to train a new classifier",
        )
        model_args.add_argument(
            "--random_proj",
            type=bool,
            default=False,
            help="whether use random projection",
        )
        model_args.add_argument(
            "--random_proj_dim",
            type=int,
            default=1024,
            help="random projection dimension",
        )
        model_args.add_argument(
            "--lr", type=float, default=0.1, help="learning rate"
        )
        model_args.add_argument(
            "--momentum",
            type=float,
            default=0.9,
            help="momentum for the optimizer",
        )
        model_args.add_argument("--left_weight", type=float, default=1)
        model_args.add_argument("--right_weight", type=float, default=1)
        model_args.add_argument("--cls_weight", type=float, default=1)
        model_args.add_argument("--mdd_weight", type=float, default=0.01)
        model_args.add_argument("--entropic_weight", type=float, default=0)
        model_args.add_argument("--loss_trade_off", type=float, default=1)
        model_args.add_argument(
            "--scheduler_lr",
            type=float,
            default=0.001,
            help="learning rate for pretrained layers",
        )
        model_args.add_argument(
            "--scheduler_weight-decay",
            type=float,
            default=0.0005,
            help="weight decay for pretrained layers",
        )
        model_args.add_argument(
            "--scheduler_gamma",
            type=float,
            default=0.001,
            help="gamma parameter for the inverse learning rate scheduler",
        )
        model_args.add_argument(
            "--scheduler_power",
            type=float,
            default=0.75,
            help="power parameter for the inverse learning rate scheduler",
        )


class RandomLayer(nn.Module):
    def __init__(self, feature_dim, num_classes, output_dim=1024):
        super(RandomLayer, self).__init__()
        self.feature_dim = feature_dim
        self.num_classes = num_classes
        self.output_dim = output_dim
        self.register_buffer("Rg", torch.randn(self.num_classes, output_dim))
        self.register_buffer("Rf", torch.randn(self.feature_dim, output_dim))

    def forward(self, feature_out, classifier_out):
        a = torch.mm(feature_out, self.Rf)
        b = torch.mm(classifier_out, self.Rg)
        return (1 / math.pow(self.output_dim, 0.5)) * torch.mul(a, b)


resnet_dict = {
    "resnet18": models.resnet18,
    "resnet34": models.resnet34,
    "resnet50": models.resnet50,
    "resnet101": models.resnet101,
    "resnet152": models.resnet152,
}


class ResNetFc(nn.Module):
    def __init__(
        self,
        resnet_name,
        use_bottleneck=True,
        bottleneck_dim=256,
        new_classifier=False,
        num_class=1000,
    ):
        super(ResNetFc, self).__init__()
        model_resnet = resnet_dict[resnet_name](pretrained=True, progress=True)
        self.feature_layers = nn.Sequential(*list(model_resnet.children())[:-1])

        self.use_bottleneck = use_bottleneck
        self.new_classifier = new_classifier
        if new_classifier:
            if self.use_bottleneck:
                self.bottleneck = nn.Linear(
                    model_resnet.fc.in_features, bottleneck_dim
                )
                self.fc = nn.Linear(bottleneck_dim, num_class)
                self.bottleneck.apply(init_weights)
                self.fc.apply(init_weights)
                self.__in_features = bottleneck_dim
            else:
                self.fc = nn.Linear(model_resnet.fc.in_features, num_class)
                self.fc.apply(init_weights)
                self.__in_features = model_resnet.fc.in_features
        else:
            self.fc = model_resnet.fc
            self.__in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.feature_layers(x)
        x = x.view(x.size(0), -1)
        if self.use_bottleneck and self.new_classifier:
            x = self.bottleneck(x)
        y = self.fc(x)
        return x, y

    def output_num(self):
        return self.__in_features

    def get_parameters(self):
        if self.new_classifier:
            if self.use_bottleneck:
                parameter_list = [
                    {
                        "params": self.feature_layers.parameters(),
                        "lr_mult": 1,
                        "decay_mult": 2,
                    },
                    {
                        "params": self.bottleneck.parameters(),
                        "lr_mult": 10,
                        "decay_mult": 2,
                    },
                    {
                        "params": self.fc.parameters(),
                        "lr_mult": 10,
                        "decay_mult": 2,
                    },
                ]
            else:
                parameter_list = [
                    {
                        "params": self.feature_layers.parameters(),
                        "lr_mult": 1,
                        "decay_mult": 2,
                    },
                    {
                        "params": self.fc.parameters(),
                        "lr_mult": 10,
                        "decay_mult": 2,
                    },
                ]
        else:
            parameter_list = [
                {"params": self.parameters(), "lr_mult": 1, "decay_mult": 2}
            ]
        return parameter_list


class AdversarialNetwork(nn.Module):
    def __init__(
        self, in_feature, hidden_size, random_layer=None, max_iter=10000
    ):
        super(AdversarialNetwork, self).__init__()
        self.ad_layer1 = nn.Linear(in_feature, hidden_size)
        self.ad_layer2 = nn.Linear(hidden_size, hidden_size)
        self.ad_layer3 = nn.Linear(hidden_size, 1)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)
        self.random_layer = random_layer
        self.apply(init_weights)
        self.iter_num = 0
        self.max_iter = max_iter

    def forward(self, x):
        if self.training:
            self.iter_num += 1
        coeff = calc_coeff(self.iter_num, max_iter=self.max_iter)
        x = GradientReverseLayer.apply(x, coeff)
        x = self.ad_layer1(x)
        x = F.relu(x)
        x = self.dropout1(x)
        x = self.ad_layer2(x)
        x = F.relu(x)
        x = self.dropout2(x)
        y = self.ad_layer3(x)
        y = torch.sigmoid(y)
        return y

    def cdan(self, feature, softmax_out, entropy=None, coeff=None):
        softmax_output = softmax_out.detach()
        if self.random_layer is None:
            op_out = torch.bmm(
                softmax_output.unsqueeze(2), feature.unsqueeze(1)
            )
            ad_out = self(
                op_out.view(-1, softmax_output.size(1) * feature.size(1))
            )
        else:
            random_out = self.random_layer(feature, softmax_output)
            ad_out = self(random_out.view(-1, random_out.size(1)))
        batch_size = softmax_output.size(0) // 2
        dc_target = torch.zeros((2 * batch_size, 1))
        dc_target = dc_target.type_as(feature)
        dc_target[:batch_size] = 1.0
        if entropy is not None:
            entropy = GradientReverseLayer.apply(entropy, coeff)
            entropy = 1.0 + torch.exp(-entropy)
            # source_weight = torch.zeros_like(entropy)
            # source_weight[: feature.size(0) // 2] = entropy[
            #     : feature.size(0) // 2
            # ]
            # target_weight = torch.zeros_like(entropy)
            # target_weight[feature.size(0) // 2 :] = entropy[
            #     feature.size(0) // 2 :
            # ]
            source_mask = torch.ones_like(entropy)
            source_mask[feature.size(0) // 2 :] = 0
            source_weight = entropy * source_mask
            target_mask = torch.ones_like(entropy)
            target_mask[0 : feature.size(0) // 2] = 0
            target_weight = entropy * target_mask
            weight = (
                source_weight / torch.sum(source_weight).detach()
                + target_weight / torch.sum(target_weight).detach()
            )
            return (
                torch.sum(
                    weight.view(-1, 1)
                    * F.binary_cross_entropy(ad_out, dc_target)
                )
                / torch.sum(weight).detach()
            )
        else:
            return F.binary_cross_entropy(ad_out, dc_target)

    def output_num(self):
        return 1

    def get_parameters(self):
        return [{"params": self.parameters(), "lr_mult": 10, "decay_mult": 2}]
