# function that prepares the text for a llama 3 instruct model
# allowing to fix the start of the assistant text with the "assistant_text" variable
# using the structure shown in https://www.llama.com/docs/model-cards-and-prompt-formats/meta-llama-3/
def prepare_text4llama3_instruct(user_text: str,
                                 system_text: str = "Assume the role of an automated system for the processing of domain-specific documentation, such as clinical or legal documents. The accuracy, robustness, consistency, and faithfulness of the reasoning performed by the system is critical in this context, and it is important to carefully consider the domain-specific terminology, to handle linguistic constructs such as temporal associations or negations, and to have robustness to different writing styles and vocabularies.",
                                 assistant_text: str = "Answer:",
                                 ):
    prompt_text = f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n{system_text}<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n{user_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n{assistant_text}"""
    return prompt_text


# tests
if __name__ == "__main__":
    res = prepare_text4llama3_instruct (user_text = "Do a trick please")
    print(res)