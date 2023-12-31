import copy
import torch
from einops import rearrange
import torch.nn as nn

from modules.dm_router import DM_Router
from modules.transformation import TPS_SpatialTransformerNetwork
from modules.feature_extraction import (
    VGG_FeatureExtractor,
    RCNN_FeatureExtractor,
    ResNet_FeatureExtractor, SVTR_FeatureExtractor,
)
from modules.sequence_modeling import BidirectionalLSTM
from modules.prediction import Attention


class Model_Extractor(nn.Module):
    def __init__(self, opt):
        super(Model_Extractor, self).__init__()
        self.opt = opt
        self.stages = {
            "Trans": opt.Transformation,
            "Feat": opt.FeatureExtraction,
            "Seq": opt.SequenceModeling,
            "Pred": opt.Prediction,
        }

        """ Transformation """
        if opt.Transformation == "TPS":
            self.Transformation = TPS_SpatialTransformerNetwork(
                F=opt.num_fiducial,
                I_size=(opt.imgH, opt.imgW),
                I_r_size=(opt.imgH, opt.imgW),
                I_channel_num=opt.input_channel,
            )
        else:
            print("No Transformation module specified")

        """ FeatureExtraction """
        if opt.FeatureExtraction == "VGG":
            self.FeatureExtraction = VGG_FeatureExtractor(
                opt.input_channel, opt.output_channel
            )
        elif opt.FeatureExtraction == "RCNN":
            self.FeatureExtraction = RCNN_FeatureExtractor(
                opt.input_channel, opt.output_channel
            )
        elif opt.FeatureExtraction == "ResNet":
            self.FeatureExtraction = ResNet_FeatureExtractor(
                opt.input_channel, opt.output_channel
            )
        elif opt.FeatureExtraction == "SVTR":
            self.FeatureExtraction = SVTR_FeatureExtractor(
                opt.input_channel, opt.output_channel
            )
        else:
            raise Exception("No FeatureExtraction module specified")
        self.FeatureExtraction_output = opt.output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d(
            (None, 1)
        )  # Transform final (imgH/16-1) -> 1

        """Sequence modeling"""
        if opt.SequenceModeling == "BiLSTM":
            self.SequenceModeling = nn.Sequential(
                BidirectionalLSTM(
                    self.FeatureExtraction_output, opt.hidden_size, opt.hidden_size
                ),
                BidirectionalLSTM(
                    opt.hidden_size, opt.hidden_size, opt.hidden_size
                ),
            )
            self.SequenceModeling_output = opt.hidden_size
        else:
            self.SequenceModeling = nn.Sequential(
                nn.Linear(
                    self.FeatureExtraction_output, opt.hidden_size)
            )
            print("No SequenceModeling module specified")
            self.SequenceModeling_output = opt.hidden_size

    def forward(self, image):
        """Transformation stage"""
        if not self.stages["Trans"] == "None":
            image = self.Transformation(image)

        """ Feature extraction stage """
        visual_feature = self.FeatureExtraction(image)
        visual_feature = visual_feature.permute(
            0, 3, 1, 2
        )  # [b, c, h, w] -> [b, w, c, h]
        visual_feature = self.AdaptiveAvgPool(
            visual_feature
        )  # [b, w, c, h] -> [b, w, c, 1]
        visual_feature = visual_feature.squeeze(3)  # [b, w, c, 1] -> [b, w, c]

        """ Sequence modeling stage """
        contextual_feature = self.SequenceModeling(
            visual_feature
        )  # [b, num_steps, opt.hidden_size]
        return contextual_feature  # [b, num_steps, opt.num_class]



class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.opt = opt
        self.model = Model_Extractor(opt)
        self.SequenceModeling_output = self.model.SequenceModeling_output
        self.stages = {
            "Pred": opt.Prediction,
        }
        self.fc = None
        self.Prediction=None

    def reset_class(self, opt, device):

        """Prediction"""
        if opt.Prediction == "CTC":
            self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == "Attn":
            self.Prediction = Attention(
                self.SequenceModeling_output, opt.hidden_size, opt.num_class
            )
        else:
            raise Exception("Prediction is neither CTC or Attn")
        
        self.Prediction.to(device)
    


    def forward(self, image, text=None, is_train=True):
        """Transformation stage"""
        contextual_feature = self.model(image)
        """ Prediction stage """
        if self.stages["Pred"] == "CTC":
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(
                contextual_feature.contiguous(),
                text,
                is_train,
                batch_max_length=self.opt.batch_max_length,
            )

        # return prediction  # [b, num_steps, opt.num_class]
        return {"predict":prediction,"feature":contextual_feature}

    def update_fc(self, hidden_size, nb_classes,device=None):
        fc = nn.Linear(hidden_size, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output] = weight
            fc.bias.data[:nb_output] = bias

        # del self.fc
        self.fc = fc

    def new_fc(self, hidden_size, nb_classes):
        # print("new_fc")
        self.fc = nn.Linear(hidden_size, nb_classes)

    def weight_align(self, increment):
        weights=self.fc.weight.data
        newnorm=(torch.norm(weights[-increment:,:],p=2,dim=1))
        oldnorm=(torch.norm(weights[:-increment,:],p=2,dim=1))
        meannew=torch.mean(newnorm)
        meanold=torch.mean(oldnorm)
        gamma=meanold/meannew
        print('alignweights,gamma=',gamma)
        self.fc.weight.data[-increment:,:]*=gamma

    def build_prediction(self,opt,num_class):
        """Prediction"""
        # print("build_prediction")
        if opt.Prediction == "CTC":
            # self.fc = nn.Linear(self.SequenceModeling_output, num_class)
            self.Prediction = self.fc
            # self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == "Attn":
            # self.fc = nn.Linear(opt.hidden_size, num_class)
            self.Prediction = Attention(
                self.SequenceModeling_output, opt.hidden_size, num_class,self.fc
            )
        else:
            raise Exception("Prediction is neither CTC or Attn")

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self



class DERNet(Model):
    def __init__(self, opt):
        super(DERNet,self).__init__(opt)
        self.model = nn.ModuleList()
        self.out_dim = None
        self.fc = None
        self.aux_fc=None
        self.task_sizes = []

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim*len(self.model)

    def extract_vector(self, x):
        features = [convnet(x) for convnet in self.model]
        features = torch.cat(features, 1)
        return features

    def forward(self, image, text=None, is_train=True):
        """Transformation stage"""
        features = [convnet(image) for convnet in self.model]
        contextual_feature = torch.cat(features, -1)

        """ Prediction stage """
        if self.stages["Pred"] == "CTC":
            prediction = self.Prediction(contextual_feature.contiguous())
        else:
            prediction = self.Prediction(
                contextual_feature.contiguous(),
                text,
                is_train,
                batch_max_length=self.opt.batch_max_length,
            )

        """ Prediction stage """
        if self.stages["Pred"] == "CTC":
            aux_logits = self.aux_Prediction(contextual_feature[:,:,-self.out_dim:].contiguous())
        else:
            aux_logits = self.aux_Prediction(
                contextual_feature[:,:,-self.out_dim:].contiguous(),
                text,
                is_train,
                batch_max_length=self.opt.batch_max_length,
            )
        # out=self.fc(features) #{logics: self.fc(features)}
        out = dict({"logits":prediction})
        # aux_logits=self.aux_fc(contextual_feature[:,-self.out_dim:])

        out.update({"aux_logits":aux_logits,"features":contextual_feature.contiguous()})
        return out  # [b, num_steps, opt.num_class]

    def update_fc(self, hidden_size, nb_classes,device=None):
        if len(self.model)==0:
            self.model.append(Model_Extractor(self.opt))
        else:
            self.model.append(Model_Extractor(self.opt))
            self.model[-1].load_state_dict(self.model[-2].state_dict())

        if self.out_dim is None:
            self.out_dim=self.model[-1].SequenceModeling_output
        fc = nn.Linear(self.feature_dim if self.opt.Prediction=="CTC" else self.out_dim, nb_classes)
        if self.fc is not None:
            nb_output = self.fc.out_features
            weight = copy.deepcopy(self.fc.weight.data)
            bias = copy.deepcopy(self.fc.bias.data)
            fc.weight.data[:nb_output,:self.feature_dim-self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc
        # new_task_size = nb_classes - sum(self.task_sizes)
        # self.task_sizes.append(new_task_size)

        self.aux_fc= nn.Linear(self.out_dim,nb_classes)

    def build_prediction(self,opt,num_class):
        """Prediction"""
        # print("build_prediction")
        if opt.Prediction == "CTC":
            # self.fc = nn.Linear(self.SequenceModeling_output, num_class)
            self.Prediction = self.fc
            # self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == "Attn":
            # self.fc = nn.Linear(opt.hidden_size, num_class)
            self.Prediction = Attention(
                self.feature_dim, opt.hidden_size, num_class,self.fc
            )
        else:
            raise Exception("Prediction is neither CTC or Attn")

    def build_aux_prediction(self,opt,num_class):
        """Prediction"""
        if opt.Prediction == "CTC":
            # self.aux_fc = nn.Linear(self.SequenceModeling_output, num_class)
            self.aux_Prediction = self.aux_fc
            # self.Prediction = nn.Linear(self.SequenceModeling_output, opt.num_class)
        elif opt.Prediction == "Attn":
            # self.aux_fc = nn.Linear(opt.hidden_size, num_class)
            self.aux_Prediction = Attention(
                self.SequenceModeling_output, opt.hidden_size, num_class,self.aux_fc
            )
        else:
            raise Exception("Prediction is neither CTC or Attn")

    def freeze_conv(self):
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()

class MRNNet(nn.Module):
    def __init__(self, opt):
        super(MRNNet, self).__init__()
        self.model = nn.ModuleList()
        self.out_dim=None
        self.fc = None
        self.opt = opt
        self.task_sizes = []
        if self.opt.FeatureExtraction == "VGG":
            self.patch = 63
        elif self.opt.FeatureExtraction == "SVTR":
            self.patch = 64
        elif self.opt.FeatureExtraction == "ResNet":
            self.patch = 65
        self.router = "dm-router"  #dm-router
        self.layer_num = 1
        self.beta = 1

    @property
    def feature_dim(self):
        if self.out_dim is None:
            return 0
        return self.out_dim*len(self.model)

    def extract_vector(self, x):
        features = [convnet(x) for convnet in self.model]
        features = torch.cat(features, 1)
        return features

    def forward(self, image, cross = True,text=None, is_train=True):
        """Transformation stage"""
        # features = [convnet(image) for convnet in self.model]
        if cross==False:
            features = self.model[-1](image,text,is_train)["predict"]
            index = None
        # elif is_train == False:
        #     features, index = self.cross_test(image)
        elif is_train == False:
            features, index = self.cross_forward_expert(image, text, is_train)
        else:
            # features,index = self.cross_forwardv2(image)
            features, index = self.cross_forward(image,text,is_train)
        # out=self.fc(features) #{logics: self.fc(features)}
        out = dict({"logits":features,"index":index,"aux_logits":None})

        return out  # [b, num_steps, opt.num_class]

    def pad_zeros_features(self,feature,total):
        B,T,know = feature.size()
        zero = torch.ones([B,T,total-know],dtype=torch.float).to(feature.device)
        return torch.cat([feature,zero],dim=-1)

    def cross_forward_expert(self, image, text=None, is_train=True):
        """Transformation stage"""
        features = [convnet(image,text,is_train) for convnet in self.model]
        route_info = torch.stack([feature["feature"] for feature in features], 1)
        route_info = self.dm_router(route_info)
        route_info = rearrange(route_info, 'b h w c -> b w (h c)')
        route_info = self.channel_route(route_info)
        # route_info = torch.cat([torch.max(feature,-1)[0] for feature in features],-1)
        index = self.route(route_info.permute(0, 2, 1).contiguous())
        # index = self.softargmax1d(torch.squeeze(index, -1),self.beta)
        index = torch.squeeze(index, -1)
        index = torch.max(index, -1)[1]

        # index [B,I]
        # route_info [B,T,I]

        # feature_array = torch.stack(features, 1)
        features = [feature["predict"] for feature in features]
        B, T, C = features[-1].size()
        list_len = len(features)
        normal_feat = []
        for i in range(list_len - 1):
            feat = self.pad_zeros_features(features[i], total=C)
            normal_feat.append(feat)
        normal_feat.append(features[-1])
        normal_feat = torch.stack(normal_feat, 0)
        # normal_feat [I,B,T,C] -> [T,C,B,I] -> [B,T,C,I]
        output = torch.stack([normal_feat[index_one][i,:,:]for i,index_one in enumerate(index)],0)

        return output.contiguous(),index

    def cross_forward(self, image, text=None, is_train=True):
        """Transformation stage"""
        features = [convnet(image,text,is_train) for convnet in self.model]
        route_info = torch.stack([feature["feature"] for feature in features], 1)
        route_info = self.dm_router(route_info)
        route_info = rearrange(route_info, 'b h w c -> b w (h c)')
        route_info = self.channel_route(route_info)
        # route_info = torch.cat([torch.max(feature,-1)[0] for feature in features],-1)
        index = self.route(route_info.permute(0, 2, 1).contiguous())
        index = self.softargmax1d(torch.squeeze(index, -1),self.beta)
        # index [B,I]
        # route_info [B,T,I]

        features = [feature["predict"] for feature in features]
        B, T, C = features[-1].size()
        list_len = len(features)
        normal_feat = []
        for i in range(list_len - 1):
            feat = self.pad_zeros_features(features[i], total=C)
            normal_feat.append(feat)
        normal_feat.append(features[-1])
        normal_feat = torch.stack(normal_feat, 0)
        # normal_feat [I,B,T,C] -> [T,C,B,I] -> [B,T,C,I]
        output = (normal_feat.permute(2, 3, 1, 0) * index).permute(2, 0, 1, 3).contiguous()
        # output = (normal_feat.permute(3,1,2,0) * route_info).permute(1,2,0,3).contiguous()

        return torch.sum(output, -1), index

    def build_fc(self, hidden_size, nb_classes):
        self.update_fc(hidden_size, nb_classes)

    def update_fc(self, hidden_size, nb_classes):
        self.model.append(Model(self.opt))
        self.model[-1].new_fc(hidden_size,nb_classes)
            # self.model[-1].load_state_dict(self.model[-2].state_dict())

        if self.out_dim is None:
            self.out_dim=self.model[-1].SequenceModeling_output


        self.route = nn.Linear(self.patch  , 1)
        self.channel_route = nn.Linear(self.feature_dim, len(self.model))
        # if self.router == "gmlp":
        #     block = GatingMlpBlock(self.out_dim, self.out_dim * 2, self.patch)
        # elif self.router == "vip":
        #     block = PermutatorBlock(self.out_dim, 2, taski = len(self.model), patch = self.patch)
        # el
        if self.router == "dm-router":
            block = DM_Router(self.out_dim, self.out_dim * 2, self.patch,len(self.model))
        else:
            block = nn.Linear(self.out_dim, self.out_dim )
        layers=[]
        for _ in range(self.layer_num):
            layers.append(block)
        print("mlp {} has {} layers".format(block, len(layers)))
        self.dm_router = nn.Sequential(*layers)
        # [b, num_steps * len] -> [b, len]
        # if self.fc is not None:
        #     nb_output = self.fc.out_features
        #     weight = copy.deepcopy(self.fc.weight.data)
        #     bias = copy.deepcopy(self.fc.bias.data)
        #     fc.weight.data[:nb_output,:self.feature_dim-self.out_dim] = weight
        #     fc.bias.data[:nb_output] = bias
        #
        # del self.fc
        # self.fc = fc
        # fc = nn.Linear(self.feature_dim, nb_classes)
    def load_fc(self,input,output):
        fc = nn.Linear(input,output)
        if self.channel_route is not None:
            nb_output = self.channel_route.out_features
            weight = copy.deepcopy(self.channel_route.weight.data)
            bias = copy.deepcopy(self.channel_route.bias.data)
            fc.weight.data[:nb_output,:self.feature_dim-self.out_dim] = weight
            fc.bias.data[:nb_output] = bias

        del self.fc
        self.fc = fc

    def build_prediction(self,opt,num_class):
        """Prediction"""
        if opt.Prediction == "CTC" or opt.Prediction == "Attn":
            # self.fc = nn.Linear(self.SequenceModeling_output, num_class)
            # self.Prediction = self.fc
            self.model[-1].build_prediction(opt,num_class)
        else:
            raise Exception("Prediction is neither CTC or Attn")

    def copy(self):
        return copy.deepcopy(self)

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()

        return self

    def softargmax1d(self,input, beta=5):
        return nn.functional.softmax(beta * input, dim=-1)


