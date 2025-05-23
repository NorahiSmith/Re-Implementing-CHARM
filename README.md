# Re-Implementing-CHARM

### Norah Smith, Max Machado, Bobby Diaz

This is the code base for Recreating CHARM: Calibrating Reward Models with Chatbot Arena Scores.  
The code is based on the paper by Xiao Zhu, Chenmien Tan, Pinzhen Chen, Rico Sennrich, Yanlin Zhang, and Hanxu Hu:  
https://arxiv.org/html/2504.10045v1#:~:text=A%20reward%20model%20assigns%20scores,the%20most%20human%2Daligned%20outputs.

## Scripts:
- The implementation of the paper's CHARM algorithm can be found in `Generate_Calibrated_Dataset_CHARM.py` 
    - This implementation relies on ELOs from Chatbot Arena: https://lmarena.ai
- `Compare_Responses.ipynb` creates comparative figures of LLM performances as scored by several different LLMs.
- `RLHF_Training.ipynb` uses the RLHF training paradigm to train a model using the generated calibrated pairwise preference dataset: https://rlhflow.github.io
- `AlpacaEval.ipynb` evaluates the trained model using the Alpaca Eval model: https://github.com/tatsu-lab/alpaca_eval
- The `datasets` folder contains the datasets generated by `Generate_Calibrated_Dataset_CHARM.py` and used to train the reward models

## Bibliography:
[1] ChatGPT, O. (2025). Screenshot of a chatgpt conversation. https://chat.openai.com. Personal screenshot taken on May 11, 2025.   
[2] Chiang, W.-L., et al. (2024). Chatbot arena: An open platform for evaluating LLMs by human preference.  
[3] Dang, J., et al. (2024). Aya Expanse: Combining research breakthroughs for a new multilingual frontier.  
[4] Dean, B. (2023). ChatGPT statistics and facts (2023). https://backlinko.com/chatgpt-stats. Accessed: 2025-05-11.  
[5] DeepMind, G. (2024a). Gemma 2 2B Instruct. https://huggingface.co/google/gemma-2-2b-it. Accessed: 2025-05-11.  
[6] DeepMind, G. (2024b). Gemma 2 9B Instruct. https://huggingface.co/google/gemma-2-9b-it. Accessed: 2025-05-11.  
[7] DeepMind, G. (2024c). Gemma 2B Instruct. https://huggingface.co/google/gemma-2b-it. Accessed: 2025-05-11.  
[8] DeepSeek-AI et al. (2025). DeepSeek-R1: Incentivizing reasoning capability in LLMs via reinforcement learning.  
[9] Dong, H., et al. (2024). RLHF Workflow: From reward modeling to online RLHF.  
[10] et al., A. G. (2024). The LLaMA 3 herd of models.  
[11] Ju, Y., & Ma, H. (2024). Training data for large language models.  
[12] Lambert, N., et al. (2024). Tülu 3: Pushing frontiers in open language model post-training.  
[13] Li, J., et al. (2016). Deep reinforcement learning for dialogue generation.  
[14] Li, X., et al. (2023). AlpacaEval: An automatic evaluator of instruction-following models. https://github.com/tatsu-lab/alpaca_eval  
[15] Microsoft (2024). Phi-3 Mini 4K Instruct. https://huggingface.co/microsoft/Phi-3-mini-4k-instruct. Accessed: 2025-05-11.  
[16] Ouyang, L., et al. (2022). Training language models to follow instructions with human feedback.  
[17] Penedo, G., et al. (2023). The RefinedWeb dataset for Falcon LLM.  
[18] Research, I. (2024). Granite-3.1-8B-Instruct. https://huggingface.co/ibm-granite/granite-3.1-8b-instruct. Accessed: 2025-05-11.  
[19] Zhu, B., et al. (2023). Starling-7B: Improving LLM helpfulness and harmlessness with RLAIF.  
[20] Zhu, X., et al. (2025). CHARM: Calibrating reward models with Chatbot Arena scores. https://arxiv.org/abs/2504.10045  
