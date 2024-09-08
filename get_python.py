import tokenize
from io import BytesIO

def get_python_types(code):
    if type(code)==list:
        code=" ".join(code)
    if type(code)==str:
        code=" ".join(code.split())

    code_bytes = code.encode('utf-8')
    code_stream = BytesIO(code_bytes)

    tokens = tokenize.tokenize(code_stream.readline)

    token_types = []
    for tok in tokens:

        if tok.type in (tokenize.ENCODING, tokenize.NEWLINE, tokenize.NL, tokenize.ENDMARKER):
            continue

        token_type_str = tokenize.tok_name[tok.type]
        token_types.append([tok.string,token_type_str])

    return [i[1] for i in token_types]


if __name__ == '__main__':

    code = """
    def add_numbers(a, b):
        result = a + b
        return result

    print(add_numbers(5, 10))
    """
    # code="""
    # public class HelloWorld {
    #     public static void main(String[] args) {
    #         System.out.println("Hello, World!");
    #     }
    # }
    # """

    result = get_python_types(code)
    print(result)

    # for word, token_type in result:
    #     print(f"Type: {token_type}, Value: {word}")
