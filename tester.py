import re

def extract_content_if_quotes(string):
    # Look for pairs of double or single quotes
    match = re.search(r'["\'](.*?)["\']', string)
    
    if match:
        # Get the content between the quotes
        content_between_quotes = match.group(1)
        
        # Count the words in the content
        word_count = len(content_between_quotes.split())
        
        # If more than 5 words, return content between quotes, otherwise return original string
        if word_count > 3:
            return content_between_quotes
    
    return string

def run_tests():
    test_cases = [
        ('This is a "simple test" string', 'This is a "simple test" string'), # less than 5 words
        ('This is a "test with more than five words here inside"', 'test with more than five words here inside'), # more than 5 words
        ("Nothing here", "Nothing here"), # no quotes
        ("This is a 'short example'", "This is a 'short example'"), # less than 5 words, single quotes
        ("Here is 'another test with more than five words inside quotes' fkorkfokrofo", 'another test with more than five words inside quotes') # more than 5 words, single quotes
    ]
    
    for i, (input_str, expected_output) in enumerate(test_cases):
        result = extract_content_if_quotes(input_str)
        assert result == expected_output, f"Test {i+1} failed: Expected {expected_output} but got {result}"
        print(f"Test {i+1} passed!")
        
run_tests()
