from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from dataclasses import dataclass, field


@dataclass
class InspectionArguments:
    model_name: str = field(
        default="none",
        metadata={"help": "HuggingFace model name"}
    )


if __name__ == "__main__":
    parser = HfArgumentParser(InspectionArguments)
    args = parser.parse_args_into_dataclasses()[0]
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model_name)
        tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    except:
        raise Exception("model does not exist, provide a huggingface model")
    
    
    
