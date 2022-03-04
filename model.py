from transformers import RobertaModel, RobertaPreTrainedModel 
from transformers import Trainer
import torch
from torch import nn

class RoBERTa_PCL(RobertaPreTrainedModel):

    def __init__(self, config):
        super().__init__(config)

        # BERT Model
        self.bert = RobertaModel(config)
        
        # Task A
        self.projection = torch.nn.Sequential(torch.nn.Dropout(0.2),
                                                torch.nn.Linear(config.hidden_size, 2))
        
        self.init_weights()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None):
 
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        # Logits A
        logits = self.projection(outputs[1])
        
        return logits
    
class MyTrainer(Trainer):
    def compute_loss(self, model, inputs):
        label = inputs.pop('label')
        outputs = model(**inputs)

        # TASK A
        loss_task = nn.CrossEntropyLoss()
        loss = loss_task(outputs.view(-1, 2), label.view(-1))
        
        return loss
