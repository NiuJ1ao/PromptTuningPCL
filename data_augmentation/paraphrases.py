from parrot import Parrot
import torch
import warnings
warnings.filterwarnings("ignore")

''' 
uncomment to get reproducable paraphrase generations
'''
def random_state(seed):
  torch.manual_seed(seed)
  if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

random_state(1234)

#Init models (make sure you init ONLY once if you integrate this to your code)
parrot = Parrot(model_tag="prithivida/parrot_paraphraser_on_T5")

phrases = ["Can you recommed some upscale restaurants in Newyork?",
           "What are the famous places we should not miss in Russia?"]


for phrase in phrases:
    print("-"*100)
    print("Input_phrase: ", phrase)
    print("-"*100)
    para_phrases = parrot.augment(input_phrase=phrases, 
                                    use_gpu=False, 
                                    diversity_ranker="levenshtein", 
                                    do_diverse=True, 
                                    max_return_phrases = 9, 
                                    max_length=100, 
                                    adequacy_threshold = 0.90, 
                                    fluency_threshold = 0.90)
    for para_phrase in para_phrases:
        print(para_phrase)