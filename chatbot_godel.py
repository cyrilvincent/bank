from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")
model = AutoModelForSeq2SeqLM.from_pretrained("microsoft/GODEL-v1_1-base-seq2seq")

def generate(instruction, knowledge, dialog):
    if knowledge != '':
        knowledge = '[KNOWLEDGE] ' + knowledge
    dialog = ' EOS '.join(dialog)
    query = f"{instruction} [CONTEXT] {dialog} {knowledge}"
    input_ids = tokenizer(f"{query}", return_tensors="pt").input_ids
    outputs = model.generate(input_ids, max_length=128, min_length=8, top_p=0.9, do_sample=True)
    output = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return output

# Instruction for a chitchat task
instruction = f'Instruction: given a dialog context, you need to response empathically.'
# Leave the knowldge empty
knowledge = ''
dialog = []
# response = generate(instruction, knowledge, dialog)
# print(response)

def chatbot():
    exit_chatbot = False
    first_loop = True

    while exit_chatbot == False:
        if (first_loop):
            print("Welcome to the chatbot! Type q to close it, otherwise let's keep chatting :)")
            print("What do you want to say?")
            first_loop = False
        print("You: ")
        user_input_question = input()

        if (user_input_question.lower() == 'q'):
            exit_chatbot = True
            print("Thank you for spending time with me, it was nice chatting with you.")
        else:
            dialog.append(user_input_question)
            response = generate(instruction, knowledge, dialog)
            print("ChatBot: \n" + response)

chatbot()