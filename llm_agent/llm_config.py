class GPTConfig:
    def __init__(self, character_id="default", stream=False, tokenizer=None, 
                 keep_dialog=None, company='openai', model='gpt-4o-mini', temperature=0.3, max_tokens_output=4096, 
                 max_tokens_context=30000, api_key_path='./settings/config.json',
                 warmed_up_dialog=None, warmup_mode=False):
        self.character_id = character_id
        self.stream = stream
        self.tokenizer = tokenizer
        self.keep_dialog = keep_dialog
        self.company = company
        self.model = model
        self.temperature = temperature
        self.max_tokens_output = max_tokens_output
        self.max_tokens_context = max_tokens_context
        self.api_key_path = api_key_path
        self.warmed_up_dialog = warmed_up_dialog
        self.warmup_mode = warmup_mode
