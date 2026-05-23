import torch
from transformers import AutoImageProcessor, AutoModel

class dino_classifier(torch.nn.Module):
    def __init__(self, 
                # model_name="facebook/dinov2-with-registers-large", 
                model_name='/data_center/data2/dataset/czj/detector/model/dinov2-l-register',
                classifier_type='linear',
                mlp_hidden_size=512
                ):
        super(dino_classifier, self).__init__()
        self.backbone = AutoModel.from_pretrained(model_name)
        if classifier_type == 'linear':
            self.fc = torch.nn.Linear(self.backbone.config.hidden_size, 1)
        elif classifier_type == 'mlp':
            self.fc = torch.nn.Sequential(
                torch.nn.Linear(self.backbone.config.hidden_size, mlp_hidden_size),
                torch.nn.ReLU(inplace=False),
                torch.nn.Dropout(0.5), 
                torch.nn.Linear(mlp_hidden_size, 1)
            )
        else:
            raise ValueError(f"Unknown classifier_type: {classifier_type}")
        
        # Freeze backbone
        # for param in self.vision_model.parameters():
        #     param.requires_grad = False
            
    def extract_feature(self, input):
        outputs = self.backbone(input)
        sequence_output = outputs[0] # batch_size, sequence_length, hidden_size
        cls_token = sequence_output[:, 0]
        return cls_token
    
    def forward(self, input, return_feature=False):
        outputs = self.backbone(input)
        sequence_output = outputs[0] # batch_size, sequence_length, hidden_size
        cls_token = sequence_output[:, 0]
        feature = cls_token
        # patch_tokens = sequence_output[:, 1:]
        # linear_input = torch.cat([cls_token, patch_tokens.mean(dim=1)], dim=1)
        logits = self.fc(cls_token)
        if return_feature:
            return logits, feature
        else:
            return logits
        