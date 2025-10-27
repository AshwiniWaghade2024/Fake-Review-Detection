from transformers import TFBertModel
import tensorflow as tf
from tensorflow.keras.layers import Layer

@tf.keras.utils.register_keras_serializable()
class BertLayer(Layer):
    def __init__(self, model_name="bert-base-uncased", **kwargs):
        self.model_name = model_name
        self.bert = TFBertModel.from_pretrained(model_name)
        super(BertLayer, self).__init__(**kwargs)

    def call(self, inputs, training=False):
        input_ids, attention_mask = inputs
        output = self.bert(input_ids, attention_mask=attention_mask)
        return output.last_hidden_state

    def get_config(self):
        config = super().get_config()
        config.update({"model_name": self.model_name})
        return config

    @classmethod
    def from_config(cls, config):
        # Handle old config saved as {"bert_model": "tf_bert_model"}
        if "bert_model" in config:
            config["model_name"] = "bert-base-uncased"  # or use logic to detect proper model
            del config["bert_model"]
        return cls(**config)