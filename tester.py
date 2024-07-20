class Create_Population:
    def __init__(self,
                 data_expanded, 
                 model, 
                 tokenizer, 
                 trie, 
                 n_samples,
                 n_pop = None,
                 history = None,
                 N=10, # for the hyper mutation thing
                 mutation_prob=0.8,
                 only_rouge = True,
                 save_preds4semeval_test = False,
                 folder = None,
                 task_w_one_shot = False,
                 task_w_highlight = False,
                 task_w_self_reasoning = False,
                 task_w_oracle_spans = False,
                 task_w_full_contract = True,
                 task_w_2_labels = True,):
        
        self.data_expanded = data_expanded
        self.model = model
        self.tokenizer = tokenizer
        self.trie = trie
        self.n_samples = n_samples
        self.n_pop = n_pop
        self.history = history
        self.N = N
        self.mutation_prob = mutation_prob
        self.only_rouge = only_rouge
        self.save_preds4semeval_test = save_preds4semeval_test
        self.folder = folder
        self.task_w_one_shot = task_w_one_shot
        self.task_w_highlight = task_w_highlight
        self.task_w_self_reasoning = task_w_self_reasoning
        self.task_w_oracle_spans = task_w_oracle_spans
        self.task_w_full_contract = task_w_full_contract
        self.task_w_2_labels = task_w_2_labels


    def __add__(self, other):
        """Overload the + operator to add the value of two MyClass instances."""
        if isinstance(other, MyClass):
            return MyClass(self.value + other.value)
        else:
            return MyClass(self.value + other)

    def copy(self):
        """Return a new instance of MyClass with the same value."""
        return MyClass(self.value)

# Example usage:
obj1 = MyClass(10)
obj2 = MyClass(5)

print("Initial value of obj1:", obj1.value)  # Output: 10
print("Initial value of obj2:", obj2.value)  # Output: 5

obj3 = obj1 + obj2
print("Value of obj3 (obj1 + obj2):", obj3.value)  # Output: 15

obj4 = obj1 + 7
print("Value of obj4 (obj1 + 7):", obj4.value)  # Output: 17

new_obj = obj1.copy()
print("Copy's value:", new_obj.value)  # Output: 10

def create_population(task, prompts_dict, initial,
                      data_expanded, 
                      model, 
                      tokenizer, 
                      trie, 
                      n_samples,
                      n_pop = None,
                      history = None,
                      N=10, # for the hyper mutation thing
                      mutation_prob=0.8,
                      only_rouge = True,
                      save_preds4semeval_test = False,
                      folder = None,
                      task_w_one_shot = False,
                      task_w_highlight = False,
                      task_w_self_reasoning = False,
                      task_w_oracle_spans = False,
                      task_w_full_contract = True,
                      task_w_2_labels = True,
                      ):