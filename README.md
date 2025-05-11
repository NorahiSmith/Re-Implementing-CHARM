# Re-Implementing-CHARM

This is the code base for Recreating CHARM: Calibrating Reward Models with Chatbot Arena Scores. The code is based on the paper by Xiao Zhu, Chenmien Tan, Pinzhen Chen, Rico Sennrich, Yanlin Zhang, and Hanxu Hu at https://arxiv.org/html/2504.10045v1#:~:text=A%20reward%20model%20assigns%20scores,the%20most%20human%2Daligned%20outputs.

Scripts:\\
    - The implementation of the paper's CHARM algorithm can be found in Generate_Calibrated_Dataset_CHARM.py\\
        - This implementation relies on ELOs from Chatbot Arena: https://lmarena.ai\\
    - Compare_Responses.ipynb creates comparative figures of LLM performances as scored by several different LLMs.\\
    - RLHF_Training.ipynb uses the RLHF training paradigm to train a model using the generated calibrated pairwise preference dataset: https://rlhflow.github.io\\
    - AlpacaEval.ipynb evaluates the trained model using the Alpaca Eval model: https://github.com/tatsu-lab/alpaca_eval\\

Bibliography:\\
[1] ChatGPT, O. (2025). Screenshot of a chatgpt conversation. https://chat.openai.com. Personal screenshot taken on May 11, 2025.\\
[2] Chiang, W.-L., Zheng, L., Sheng, Y., Angelopoulos, A. N., Li, T., Li, D., Zhang, H., Zhu, B., Jordan, M., Gonzalez, J. E., and Stoica, I. (2024). Chatbot arena: An open platform for evaluating llms by human preference.\\
[3] Dang, J., Singh, S., D’souza, D., Ahmadian, A., Salamanca, A., Smith, M., Peppin, A., Hong, S., Govindassamy, M., Zhao, T., Kublik, S., Amer, M., Aryabumi, V., Campos, J. A., Tan, Y.-C., Kocmi, T., Strub, F., Grinsztajn, N., Flet-Berliac, Y., Locatelli, A., Lin, H., Talupuru, D., Venkitesh, B., Cairuz, D., Yang, B., Chung, T., Ko, W.-Y., Shi, S. S., Shukayev, A., Bae, S., Piktus, A., Castagné, R., Cruz-Salinas, F., Kim, E., Crawhall-Stein, L., Morisot, A., Roy, S., Blunsom, P., Zhang, I., Gomez, A., Frosst, N., Fadaee, M., Ermis, B., Üstün, A., and Hooker, S. (2024). Aya expanse: Combining research breakthroughs for a new multilingual frontier.\\
[4] Dean, B. (2023). Chatgpt statistics and facts (2023). https://backlinko.com/chatgpt-stats. Accessed: 2025-05-11.\\
[5] DeepMind, G. (2024a). Gemma 2 2b instruct. https://huggingface.co/google/gemma-2-2b-it. Accessed: 2025-05-11.\\
[6] DeepMind, G. (2024b). Gemma 2 9b instruct. https://huggingface.co/google/gemma-2-9b-it. Accessed: 2025-05-11.\\
[7] DeepMind, G. (2024c). Gemma 2b instruct. https://huggingface.co/google/gemma-2b-it. Accessed: 2025-05-11.\\
[8] DeepSeek-AI and et al, D. G. (2025). Deepseek-r1: Incentivizing reasoning capability in llms via
reinforcement learning.\\
[9] Dong, H., Xiong, W., Pang, B., Wang, H., Zhao, H., Zhou, Y., Jiang, N., Sahoo, D., Xiong, C., and Zhang, T. (2024). Rlhf workflow: From reward modeling to online rlhf.\\
[10] et al, A. G. (2024). The llama 3 herd of models.\\
[11] Ju, Y. and Ma, H. (2024). Training data for large language model.\\
[12] Lambert, N., Morrison, J., Pyatkin, V., Huang, S., Ivison, H., Brahman, F., Miranda, L. J. V., Liu, A., Dziri, N., Lyu, S., Gu, Y., Malik, S., Graf, V., Hwang, J. D., Yang, J., Bras, R. L., Tafjord, O., Wilhelm, C., Soldaini, L., Smith, N. A., Wang, Y., Dasigi, P., and Hajishirzi, H. (2024). Tülu 3: Pushing frontiers in open language model post-training.\\
[13] Li, J., Monroe, W., Ritter, A., Galley, M., Gao, J., and Jurafsky, D. (2016). Deep reinforcement learning for dialogue generation.\\
[14] Li, X., Zhang, T., Dubois, Y., Taori, R., Gulrajani, I., Guestrin, C., Liang, P., and Hashimoto, T. B. (2023). Alpacaeval: An automatic evaluator of instruction-following models. https://github. com/tatsu-lab/alpaca_eval.\\
[15] Microsoft (2024). Phi-3 mini 4k instruct. https://huggingface.co/microsoft/ Phi-3-mini-4k-instruct. Accessed: 2025-05-11.\\
[16] Ouyang, L., Wu, J., Jiang, X., Almeida, D., Wainwright, C. L., Mishkin, P., Zhang, C., Agarwal, S., Slama, K., Ray, A., Schulman, J., Hilton, J., Kelton, F., Miller, L., Simens, M., Askell, A., Welinder, P., Christiano, P., Leike, J., and Lowe, R. (2022). Training language models to follow instructions with human feedback.\\
[17] Penedo, G., Malartic, Q., Hesslow, D., Cojocaru, R., Cappelli, A., Alobeidli, H., Pannier, B., Almazrouei, E., and Launay, J. (2023). The refinedweb dataset for falcon llm: Outperforming curated corpora with web data, and web data only.\\
[18] Research, I. (2024). Granite-3.1-8b-instruct. https://huggingface.co/ibm-granite/granite-3. 1-8b-instruct. Accessed: 2025-05-11.\\
[19] Zhu, B., Frick, E., Wu, T., Zhu, H., and Jiao, J. (2023). Starling-7b: Improving llm helpfulness harmlessness with rlaif.\\
[20] Zhu, X., Tan, C., Chen, P., Sennrich, R., Zhang, Y., and Hu, H. (2025). Charm: Calibrating reward models with chatbot arena scores. https://arxiv.org/abs/2504.10045.\\