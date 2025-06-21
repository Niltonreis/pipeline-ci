def soma(a, b):
    return a + b

def divide(a, b):
    if b == 0:
        raise ValueError("Não pode dividir por zero!")
    return a / b