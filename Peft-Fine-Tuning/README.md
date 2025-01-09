# Dialogue Summarization with PEFT Fine-tuning

This project demonstrates how to use Parameter Efficient Fine-Tuning (PEFT) to adapt a FLAN-T5 model for dialogue summarization using the DialogSum dataset.

## Features

- PEFT/LoRA fine-tuning of FLAN-T5 model
- Dialogue summarization inference
- ROUGE metric evaluation
- Python 3.9.2 or higher

## Installation

```bash
pip install --upgrade pip #Optional
pip install -r requirements.txt
```

## Training

To fine-tune your own PEFT adapter:

- To get help on model paramters, run: `python src/train.py --help` command

```bash
python src/train.py \
    --base_model "google/flan-t5-base" \
    --dataset "knkarthick/dialogsum" \
    --output_dir "./peft_model" \
    --num_epochs 1
```

## Model Details

- Base Model: FLAN-T5
  - [https://huggingface.co/google/flan-t5-base](https://huggingface.co/google/flan-t5-base)
- PEFT Method: LoRA
- Dataset: DialogSum
  - [https://huggingface.co/datasets/knkarthick/dialogsum](https://huggingface.co/datasets/knkarthick/dialogsum)
- LoRa Configuration:
  - Rank: 32
  - Alpha: 32
  - Target Modules: ["q", "v"]
  - Dropout: 0.05
- Train details
  - Total params: $251116800$ 
  - Trainable params: $3538944$
    -  $\rightsquigarrow 1.4$% of total parameters.
  - Performances:
    - ROUGE scores computed on 100 random samples from test set:
      |           |Rouge1 |Rouge2 |RougeL |
      |-----------|-------|-------|-------|
      |Base model |0.2511 |0.0917 | 0.2187|
      |Peft model |0.4006 |0.1660 | 0.3239|


    ```

## Evaluation
  
- Run `offline_evaluation.py`  for model assement, using ROUGE-L metric

## Inference

- Run `inference.py` script
- Then go to `http://127.0.0.1:7860` for model testing.
  
- Examples of dialogues
  - **Example 1**
    ```
    #Person1#: What's the matter with this computer?
    #Person2#: I don't know, but it just doesn't work well. Whenever I start it, it stops running.
    #Person1#: Have you asked Mr. Li for some advice?
    #Person2#: Yes, I have, but he doesn't seem to be able to solve the problem, either. Can you help me?
    #Person1#: Me? I know nothing more than playing computer games.
    #Person2#: What shall I do? I have to finish this report this afternoon, but...
    #Person1#: But why don't you ring up the repairmen? They will be able to settle the problem.
    #Person2#: Yes, I'll ring them up.
    ```
  - **Example 2**
    ```
    #Person1#: I am sorry, sir. I have broken the reading lamp in my room.
    #Person2#: Well, sir. May I have your room number?
    #Person1#: 503. I would like to pay for it.
    #Person2#: Please fill out the form first.
    #Person1#: OK, can you bring me a new one?
    #Person2#: Of course.
    ```
  - Example
    ```
    #Person1#: Brian, do you know how to speak English?
    #Person2#: Yes.
    #Person1#: Where did you learn?
    #Person2#: I learned in college.
    #Person1#: You speak really well.
    #Person2#: Thank you.
    #Person1#: How long have you been in the U. S.?
    #Person2#: 3 weeks.
    #Person1#: Is your wife with you?
    #Person2#: Yes, she just got here yesterday.
    #Person1#: Have you been to California before?
    #Person2#: No. I've never been there.
    #Person1#: Have you ever been to Las Vegas?
    #Person2#: Yes. I went there once on a business trip.
  ```

## References

- [Scaling Down to Scale UP: A Guide to Parameter-Efficient Fine-Tuning](https://arxiv.org/pdf/2303.15647)
- [On the Effectiveness of Parameter-Efficient Fine-Tuning](https://arxiv.org/pdf/2211.15583)
- [PEFT - Hugging Face](https://github.com/huggingface/peft)
- [https://www.coursera.org/learn/generative-ai-with-llms/](https://www.coursera.org/learn/generative-ai-with-llms/)
- [https://www.gradio.app/](https://www.gradio.app/)
