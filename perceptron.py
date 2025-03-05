import numpy as np

# Variáveis Globais
bias = 1
taxa_de_aprendizado = 0.5

def activation_func(V):
    return 1 if V > 0 else 0

def sigmoid_func(V):
    return 1 / (1 + np.exp(-taxa_de_aprendizado * V))

def calculate_error(Y_desejado, Y):
    return Y_desejado - Y

def calculate_delta(erro, X_train):
    return taxa_de_aprendizado * erro * X_train

def reconhecer_digito(X_novo, W_i, W_j):
    # Forward propagation
    v_i = np.dot(X_novo, W_i)  # Camada oculta
    y_i = activation_func(v_i)
    
    entrada_j = np.array([bias, y_i])  # Entrada para camada j (com bias)
    resultado = []
    
    for neuronio in range(2):  # Camada de saída
        v_j = np.dot(entrada_j, W_j[neuronio])
        y_j = activation_func(v_j)
        resultado.append(y_j)
    
    return resultado

def treinar_rede():
    num_epocas = 100
    limite_erro = 0.001
    num_amostras = int(input("Digite a quantidade de amostras de treinamento: "))

    X_train = []
    Y_desejado = []

    for i in range(num_amostras):
        print(f"\nAmostra {i+1}:")
        amostra = list(map(int, input("Digite a amostra de treinamento: ").split()))
        digito = int(input("Digite o dígito desejado (0 ou 1): "))
        X_train.append([bias] + amostra)
        Y_desejado.append([1, 0] if digito == 0 else [0, 1])
    
    X_train = np.array(X_train, dtype=np.float64)
    Y_desejado = np.array(Y_desejado, dtype=np.float64)

    W_i = np.random.randn(len(X_train[0]))  # Pesos da camada oculta
    W_j = np.random.randn(2, 2)  # Pesos da camada de saída (2 neurônios, 2 entradas cada)
    
    for epoca in range(num_epocas):
        erro_total = 0
        
        for i in range(len(X_train)):
            v_i = np.dot(X_train[i], W_i)
            y_i = sigmoid_func(v_i)
            print(y_i)
            
            entrada_j = np.array([bias, y_i])
            y = []
            e = []
            
            for neuronio in range(2):
                v_j = np.dot(entrada_j, W_j[neuronio])
                y_j = activation_func(v_j)
                y.append(y_j)
                e.append(calculate_error(Y_desejado[i, neuronio], y_j))
            
            deltaW_j = np.array([calculate_delta(e[n], entrada_j) for n in range(2)])
            W_j += deltaW_j
            
            erro_oculto = np.dot(e, W_j[:, 1])
            deltaW_i = calculate_delta(erro_oculto, X_train[i])
            W_i += deltaW_i
            
            erro_total += sum(e) ** 2
        
        erro_medio = erro_total / (2 * len(X_train))
        print(f"Erro médio da época {epoca+1}: {erro_medio:.6f}")
        
        if erro_medio <= limite_erro:
            print("Treinamento encerrado.")
            break
    
    np.savetxt('pesos_oculta.txt', W_i, fmt='%.2f')
    np.savetxt('pesos_saida.txt', W_j, fmt='%.2f')

def reconhecer_digito_usuario():
    X_novo = [bias] + list(map(int, input("Digite os valores da amostra: ").split()))
    W_i = np.loadtxt('pesos_oculta.txt')
    W_j = np.loadtxt('pesos_saida.txt')
    resultado = reconhecer_digito(X_novo, W_i, W_j)
    print("Dígito reconhecido:", "Zero" if resultado == [1, 0] else "Um" if resultado == [0, 1] else "Não reconhecido")

def main():
    while True:
        print("\nMenu:\n1 - Treinar\n2 - Reconhecer\n3 - Sair")
        opcao = input("Escolha: ")
        if opcao == '1': treinar_rede()
        elif opcao == '2': reconhecer_digito_usuario()
        elif opcao == '3': break
        else: print("Opção inválida!")

if __name__ == "__main__":
    main()
