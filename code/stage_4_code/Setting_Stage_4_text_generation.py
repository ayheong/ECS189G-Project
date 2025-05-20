from code.base_class.setting import setting
from Dataset_TextGeneration import Dataset_TextGeneration

class Setting_Stage_4_text_generation(setting):
    def __init__(self, sName='Stage 4 Joke Generation', sDescription=None):
        super().__init__(sName, sDescription)
        self.vocab = None
        self.prompt_words = None

    def prepare(self, sDataset, sMethod, sResult, sEvaluate=None, vocab=None, prompt_words=None):
        self.dataset = sDataset
        self.method = sMethod
        self.result = sResult
        self.evaluate = sEvaluate
        self.vocab = vocab
        self.prompt_words = prompt_words

    def load_run_save_evaluate(self):
        loaded_data = self.dataset.load()
        train_dataset = Dataset_TextGeneration(data=loaded_data['train'], vocab=self.vocab)

        self.method.train(train_dataset)

        print("\nGenerating with prompt:", self.prompt_words)
        generated_text = self.method.generate(self.prompt_words, self.vocab)

        print("\nGenerated Joke:")
        print(generated_text)

        if self.result:
            self.result.data = generated_text
            self.result.save()

        return generated_text, None

