from transformers import AutoImageProcessor, ViTModel
import torch.nn as nn

class ViT(nn.Module):
        def __init__(self, path_model_pretrained, num_tasks, num_diseases):
                super(ViT, self).__init__()
                self.num_tasks = num_tasks
                self.num_diseases = num_diseases
                self.path_model_pretrained = path_model_pretrained
                self.base_model = ViTModel.from_pretrained(path_model_pretrained, output_hidden_states=False)
                self.ent = nn.Sequential(
                        nn.Linear(768, num_tasks)
                )
                self.desease = nn.Sequential(
                        nn.Linear(768, num_diseases)
                )
                self.drop = nn.Dropout(0.4)

        def forward(self, x):
                # out = self.base_vit(x).last_hidden_state[:, 0, :]
                out = self.base_model(x).pooler_output
                # out = self.drop(out)
                task_1 = self.ent(out)
                task_2 = self.desease(out)
                return task_1, task_2

        def get_hidden_state(self, x):
                out = self.base_model(x)
                return out.hidden_states

def ViT_Multitask(config, anotations):
    path_model_pretrained = config['model_pretrained']
    num_tasks = anotations['num_tasks']
    num_disease = anotations['num_diseases']
    model = ViT(path_model_pretrained, num_tasks, num_disease)
    return model

