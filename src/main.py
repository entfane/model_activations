from transformers import AutoModelForCausalLM, AutoTokenizer, HfArgumentParser
from dataclasses import dataclass, field
from utils import ActivationAnalyzer


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
    
    activation_analyzer = ActivationAnalyzer()
    activation_analyzer.attach_forward_hooks(model)
    input = tokenizer("some input", return_tensors="pt").to(model.device)
    model.generate(**input, max_new_tokens = 1)
    print(activation_analyzer.activations)
    
    
