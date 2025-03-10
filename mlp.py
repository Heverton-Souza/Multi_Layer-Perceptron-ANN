import numpy as np

# Variáveis globais
bias = 1
taxa_de_aprendizado = 0.5

def sigmoid(V):
    return 1 / (1 + np.exp(-taxa_de_aprendizado * V))

def derivada_sigmoid(V):
    return taxa_de_aprendizado * V * (1 - V)

def calcular_erro(Y_desejado, Y):
    return Y_desejado - Y

def treinar_rede():
    num_epocas = 10000
    limite_erro = 0.001
    
    num_amostras = int(input("Digite a quantidade de amostras de treinamento: "))
    X_train, Y_desejado = [], []
    
    for i in range(num_amostras):
        print(f"\nAmostra {i+1}:")
        amostra = list(map(int, input("Digite a amostra de treinamento (valores separados por espaço): ").split()))
        digito = int(input("Digite o dígito desejado (0 ou 1): "))
        
        X_train.append([bias] + amostra)
        Y_desejado.append([1, 0] if digito == 0 else [0, 1])
    
    X_train = np.array(X_train, dtype=np.float64)
    Y_desejado = np.array(Y_desejado, dtype=np.float64)
    
    W_oculta = np.random.randn(len(X_train[0]))  # Pesos da camada oculta
    W_saida = np.random.randn(2, 2)  # Pesos da camada de saída
    
    for epoca in range(num_epocas):
        erro_total = 0
        
        for i in range(len(X_train)):
            # Forward Pass
            V_oculta = np.dot(X_train[i], W_oculta)
            Y_oculta = sigmoid(V_oculta)
            
            V_saida = np.dot([1, Y_oculta], W_saida.T)
            Y_saida = sigmoid(V_saida)
            
            erro = calcular_erro(Y_desejado[i], Y_saida)
            Ei = np.sum(erro ** 2) / len(W_saida)  # Cálculo do Ei
            erro_total += Ei
            
            # Backpropagation
            delta_saida = erro * derivada_sigmoid(Y_saida)
            delta_oculta = derivada_sigmoid(Y_oculta) * np.dot(delta_saida, W_saida[:, 1])
            
            # Atualização dos pesos
            W_saida += taxa_de_aprendizado * np.outer(delta_saida, [1, Y_oculta])
            W_oculta += taxa_de_aprendizado * delta_oculta * X_train[i]
            
        E = erro_total / len(X_train)  # Cálculo do erro médio E
        print(f"\nÉpoca {epoca+1}, Erro médio: {E:.6f}")
        
        if E <= limite_erro:
            print("Treinamento concluído!")
            break
    
    np.savetxt('pesos_oculta.txt', W_oculta)
    np.savetxt('pesos_saida.txt', W_saida)

def reconhecer_digito_usuario():
    W_oculta = np.loadtxt('pesos_oculta.txt')
    W_saida = np.loadtxt('pesos_saida.txt')
    
    amostra = list(map(int, input("\nDigite os valores da amostra: ").split()))
    X_novo = np.array([bias] + amostra, dtype=np.float64)
    
    V_oculta = np.dot(X_novo, W_oculta)
    Y_oculta = sigmoid(V_oculta)
    
    V_saida = np.dot([1, Y_oculta], W_saida.T)
    Y_saida = sigmoid(V_saida)
    
    resultado = np.argmax(Y_saida)
    print(f"\nDígito reconhecido: {resultado}")

def main():
    while True:
        print("\nMenu:")
        print("1 - Treinar a rede neural")
        print("2 - Reconhecer um dígito")
        print("3 - Sair")
        
        opcao = input("Escolha uma opção: ")
        if opcao == '1':
            treinar_rede()
        elif opcao == '2':
            reconhecer_digito_usuario()
        elif opcao == '3':
            break
        else:
            print("Opção inválida!")

if __name__ == "__main__":
    main()
