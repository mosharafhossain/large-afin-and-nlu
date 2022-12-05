# -*- coding: utf-8 -*-
"""
@author: Md Mosharaf Hossain
"""

class Config():
    def __init__(self, params):
        
        # seed and GPU setting
        self.seed               = params.pop("seed", 42)        
        self.train_shuffle      = params.pop("train_shuffle", True)
        self.use_gpu            = params.pop("use_gpu", True)
        self.device             = params.pop("device", 0)
        self.use_multi_gpus     = params.pop("use_multi_gpus", False)	
        
        # Utilizing large-scale-afin
        self.with_weak_labels     = params.pop("with_weak_labels", True)
        self.large_afin_path      = params.pop("large_afin_path",  "")
        self.blended_training     = params.pop("blended_training", True)
        self.blended_start_epoch  = params.pop("blended_start_epoch", 4)
        self.blended_end_epoch    = params.pop("blended_end_epoch", 8)
        self.alpha                = params.pop("alpha", 0.25)
    	
        
        # network params
        self.learning_rate        = params.pop("learning_rate", 1e-5)
        self.optimizer            = params.pop("optimizer", "Adafactor")
        self.adam_epsilon         = params.pop("adam_epsilon", 1e-8) 
        self.weight_decay         = params.pop("weight_decay", 0.0) 
        self.warmup_epoch         = params.pop("warmup_epoch", 5) 
        self.accumulate_step      = params.pop("accumulate_step", 1)
        self.grad_clipping        = params.pop("grad_clipping", 5.0)        
        
        self.max_epoch            = params.pop("max_epoch", 1)
        self.epochs_no_eval       = params.pop("epochs_no_eval", 0)
        self.patience             = params.pop("patience", 50)
        self.batch_size           = params.pop("batch_size", 4)
    	
        # input and target
        self.input_len            = params.pop("input_len", {})
        self.target_len           = params.pop("target_len", {})
    	
        # T5 related params
        self.num_beams            = params.pop("num_beams", None)
        self.do_sample            = params.pop("do_sample", True)
        self.early_stopping       = params.pop("early_stopping", True)
        self.top_k                = params.pop("top_k", 50)
        self.top_p                = params.pop("top_p", 0.95)
        self.repetition_penalty   = params.pop("repetition_penalty", 2.5)
        self.num_return_sequences = params.pop("num_return_sequences", 1)      
    	
        # best model and predicted results
        self.T5_type              = params.pop("T5_type", "large")
        self.T5_path              = params.pop("T5_path",  {})         
        self.is_model_tuning      = params.pop("is_model_tuning", True)
        self.data_path            = params.pop("data_path", {})
        self.output_path          = params.pop("output_path", {} )
        self.best_model_path      = params.pop("best_model_path")
        self.bad_words            = params.pop("bad_words", ["not", "n't", "no", "never", "without", "nothing", "none", "nobody", "nowhere", "neither", "nor"])
        
        assert len(params) == 0  #there should be no keys after initialize the parameters
        
        