#!/usr/bin/env python
# coding: utf-8

# In[1]:


import warnings
warnings.filterwarnings("ignore")

import numpy as np #Manipulação de vetores, matrizes, operações, etc.
import pandas as pd #Visualização, organização nos DataFrames (planilhas)
import tensorflow as tf #Machine Learning e afins
import matplotlib.pyplot as plt #Plotagem e visualização

from tensorflow import keras #API de machine learning
from keras import Sequential #Compilador p/ RNA
from keras.layers import Dense #Construção das camadas
from keras.optimizers import SGD

from sklearn import svm #SVM padrão
from sklearn.svm import SVC  
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score #Model tunning
from sklearn.preprocessing import StandardScaler #Estandardização por escalonamento
from sklearn.feature_selection import RFECV #Seleção por eliminação recursiva de atributos
from sklearn.metrics import confusion_matrix #Matriz de confusão p/ discriminar os erros e acertos na rede neural

from itertools import product
sc = StandardScaler()
plt.rcParams["figure.figsize"]=14,14
get_ipython().run_line_magic('matplotlib', 'qt')


# In[3]:


#Funções referentes à SVM
#Variáveis:
#--------- x_data, y_data - vetores de entrada e vetor de rótulos de estabilidade; C - parâmetro de regularização da SVM;
#--------- kernel - função base da SVM; cv - número de dobras para validação cruzada do Stratified K-Fold no método RFECV;
#--------- subset - dicionário em que os subconjuntos D, PG, QG e VM estão organizados;
#'svm_rfecv()' aplica o método RFECV aos vetores de entrada x_data, selecionando os atributos para o treinamento e teste da SVM

def svm_rfecv(x_data, y_data, C, kernel, cv):
    #Aplica-se a função 'RFECV_selected()'
    selected_data, selected_features = RFECV_selected(x_data, y_data, kernel, C, cv)
    
    #Divide-se os dados em conjuntos de treinamento e teste e são misturados aleatoriamente. 30% dos dados são para teste 
    x_train, x_test, y_train, y_test = train_test_split(sc.fit_transform(selected_data), y_data, test_size=0.3, shuffle = True)
    
    #Instância de uma SVM com C e kernel referentes à entrada que então adequa-se ao conjunto de treino
    clf_svm_rfecv = SVC(C=C, kernel=kernel)
    clf_svm_rfecv.fit(x_train, y_train)
    
    #Retorna precisão de treino e teste e as variáveis selecionadas no método RFECV
    return clf_svm_rfecv.score(x_train, y_train), clf_svm_rfecv.score(x_test, y_test), selected_features
#===============================================================================================================================
#'RFECV_selected()' função para aplicar RFECV e retornar os dados selecionados e os índices das variáveis
def RFECV_selected(x_data, y_data, kernel, C, cv):
    #Estimador em que o RFECV baseia a métrica de avaliação das variáveis
    estimador = SVC(kernel=kernel, C=C)
    
    #Aplicação do selecionador RFECV, com remoção de 1 atributo para cada iteração do método
    selecionador = RFECV(estimador, step=1, cv=StratifiedKFold(cv))
    selecionador = selecionador.fit(sc.fit_transform(x_data.values), y_data)
    
    #Retorna os atributos selecionados e os índices da seleção
    return x_data[x_data.columns[selecionador.get_support()]], x_data.columns[selecionador.get_support()]
#===============================================================================================================================
#'svm()' treina e testa uma SVM
def svm_std(x_data, y_data, C, kernel):
    #Divide-se os dados em conjuntos de treinamento e teste e são misturados aleatoriamente. 30% dos dados são para teste 
    x_train, x_test, y_train, y_test = train_test_split(sc.fit_transform(x_data), y_data, test_size=0.3, shuffle = True)
    
    #Instância de uma SVM com C e kernel referentes à entrada que então adequa-se ao conjunto de treino
    clf_svm = SVC(C=C, kernel=kernel)
    clf_svm.fit(x_train, y_train)
    
    #Retorna precisão de treino e teste
    return clf_svm.score(x_train, y_train), clf_svm.score(x_test, y_test)
#===============================================================================================================================
#'saida_svm_rfecv()' aplica a função 'svm_rfecv()' para C=[10, 1, 0.1, 0.01, 0.001] em todos os subconjuntos de variáveis 
def saida_svm_rfecv(subset, y_data, kernel, cv):
    C=[10, 1, 0.1, 0.01, 0.001]
    
    teste=list()
    treino=list()
    
    for i in range(5):
        acc_teste=list()
        acc_treino=list() 
        
        for conjuntos in subset: #Para cada índice de conjunto dentro de 'subset'
            print('SVM de kernel {} com seleção de atributos por RFECV com {} dobras - Conjunto: {}, C = {}, \n'.format(kernel, cv, conjuntos, C[i]))
            
            x_data = subset[str(conjuntos)]
            #Aplicação da função 'svm_rfecv()'
            treino_svm_rfecv, teste_svm_rfecv, selected_features = svm_rfecv(x_data, y_data, C[i], kernel, cv)
            
            print('Precisão de Treino: {}% \nPrecisão de Teste: {}%'.format(treino_svm_rfecv*100, teste_svm_rfecv*100))
            
            #Armazena os resultados do conjunto
            acc_teste.append(teste_svm_rfecv)
            acc_treino.append(treino_svm_rfecv)
            
            print('Conjunto {} concluído.\n'.format(conjuntos))
            print('Atributos escolhidos:')
            print(*selected_features, sep = ", ") 
            print('--'*50)
        
        print('Sequência concluída para C = {}'.format(C[i]))
        print('--'*50)
        
        #Armazena os resultados de todos os conjuntos para dado C
        teste.append(acc_teste)
        treino.append(acc_treino)

    return teste, treino
#===============================================================================================================================
#'saida_svm()' aplica a função 'svm()' para C=[10, 1, 0.1, 0.01, 0.001] em todos os conjuntos de variáveis em 'subset'
def saida_svm(subset, y_data, kernel):
    C=[10, 1, 0.1, 0.01, 0.001]
    
    teste=list()
    treino=list()
    
    for i in range(5):
        acc_teste=list()
        acc_treino=list() 
        for conjuntos in subset:
            print('SVM de kernel {} - Conjunto: {}, C = {}, \n'.format(kernel, conjuntos, C[i]))
            
            x_data = subset[str(conjuntos)]
            treino_svm, teste_svm = svm_std(x_data, y_data, C[i], kernel)
            
            print('Precisão de Treino: {}% \nPrecisão de Teste: {}%'.format(treino_svm*100, teste_svm*100))
            
            acc_teste.append(teste_svm)
            acc_treino.append(treino_svm)
            
            print('Conjunto {} concluído.\n'.format(conjuntos))

            print('--'*50)
        
        print('Sequência concluída para C = {}'.format(C[i]))
        print('--'*50)
        teste.append(acc_teste)
        treino.append(acc_treino)
    
    return teste, treino
#===============================================================================================================================
#'RFECV_test()' aplica RFECV para C = [10, 1, 0.1, 0.01, 0.001] e cv = [2, 3, 4, 5] combinados, retornando uma matriz com 
#o número de atributos selecionados e o ranking de atributos para cada C e cv
def RFECV_test(x_data,y_data):
    C = [10, 1, 0.1, 0.01, 0.001] #i
    cv = [2, 3, 4, 5] #k
    n=[]
    selected_features=[]
    c_out=[]
    cv_out=[]
    
    for i in range(5):
        #Atribui-se C = [10, 1, 0.1, 0.01, 0.001] em C[i]
        estimador = SVC(kernel='linear', C=C[i])
        
        for k in range(4):
            #Atribui-se cv = [2, 3, 4, 5] em cv[k]
            selecionador = RFECV(estimador, step=1, cv=StratifiedKFold(cv[k]))
            selecionador = selecionador.fit(sc.fit_transform(x_data.values), y_data)
            
            #selecionador.get_support() são os índices dos atributos escolhidos
            x_features = pd.DataFrame(x_data.columns[selecionador.get_support()])
            
            selected_features.append(x_features.T.iloc[0])
            
            n.append(x_features.T.iloc[0].size)
            c_out.append(C[i])
            cv_out.append(cv[k])
    rfecv = pd.DataFrame({'Número de Atributos Selecionados': n, 
                          'Ranking de Atributos Selecionados': selected_features, 
                          'C': c_out, 'Dobras Stratified K-Fold': cv_out})
    return rfecv.sort_values('Número de Atributos Selecionados')
#===============================================================================================================================
def cv_score(conjuntos, y_data, cv, figure):
    max_score = list()
    for conjunto in conjuntos:
        x_data = sc.fit_transform(conjuntos[str(conjunto)])
        estimador = SVC(kernel='linear')
        C_range = np.logspace(-10, 1, 12)
        
        pontuacao = list()
        pontuacao_std = list()

        for C in C_range:
            estimador.C = C
            c_pontuacao = cross_val_score(estimador, x_data, y_data, cv=cv, n_jobs=-1)
            pontuacao.append(np.mean(c_pontuacao))
            pontuacao_std.append(np.std(c_pontuacao))
            
        max_score.append([conjunto, np.array(pontuacao).max(), C_range[np.array(pontuacao).argmax()], cv])
        
        if figure:
            plt.figure()
            plt.semilogx(C_range, pontuacao)
            plt.semilogx(C_range, np.array(pontuacao) + np.array(pontuacao_std), 'b--')
            plt.semilogx(C_range, np.array(pontuacao) - np.array(pontuacao_std), 'b--')
            locs, labels = plt.yticks()
            plt.yticks(locs, list(map(lambda x: "%g" % x, locs)))
            plt.title('Pontuação de Validação Cruzada do Subconjunto {}'.format(conjunto))
            plt.ylabel('Score')
            plt.xlabel('C')
            plt.ylim(0, 1.1)
            plt.tight_layout()
            plt.show()
            
    return pd.DataFrame(max_score, columns=['Subconjunto', 'Pontuação', 'C', 'Dobras'])
#===============================================================================================================================        
#Função de construção da arquitetura da RNA
#Entradas da função: número de camadas, arquitetura, função de ativação, tamanho do vetor de entrada e número de saídas.
def build_model(n_layers, architecture, act_func, input_shape, output):
    model = Sequential() #Sequential do Keras é um modelo de camadas lineares de neurônios, que permite a estruturação de
                         #redes entre as camadas.
    model.add(Dense(architecture[0], input_dim=input_shape)) #Camada de entrada, passa-se arquitetura e tamanho do vetor de entrada
    
    #Seleciona-se a função de ativação dos neurônios  - unidade linear retificada, função sigmoide ou função tangente hiperbólica
    if act_func=='relu': 
        activation=tf.nn.relu
    elif act_func=='sigmoid':
        activation=tf.nn.sigmoid
    elif act_func=='tanh':
        activation=tf.nn.tanh
    
    for i in range(n_layers): #Estrutura da arquitetura das camadas internas    
        model.add(Dense(architecture[i], activation=activation))
    model.add(Dense(output, activation='sigmoid')) #Camada de saída
    return model #Retorna o modelo construído com as entradas da função build_model()
#===============================================================================================================================

def compile_train_model(model, x_train, y_train, lr, batch_size, epochs, verbose, callbacks):
    model_copy = model
    
    model_copy.compile(optimizer=SGD(lr=lr),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    
    model_copy.fit(x_train, y_train, epochs=epochs, validation_split=0.175, batch_size=batch_size
                   , verbose=verbose, callbacks=callbacks)
    return model_copy
#===============================================================================================================================
def plot_loss_acc(model, target_acc, title, tag1, tag2):
    e=np.array(model.history.epoch)+1
    l=np.array(model.history.history['loss'])
    a=np.array(model.history.history['acc'])
    
    fig, ax1 = plt.subplots()

    plt.rcParams.update({'font.size': 13})
    plt.title(title)
    color = 'tab:blue'
    
    ax1.set_xlabel('Época')
    ax1.set_ylabel('Precisão do Modelo', color=color)
    ax1.plot(e, a, color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

    color = 'tab:red'
    ax2.set_ylabel('Perda do Modelo', color=color)  # we already handled the x-label with ax1
    ax2.plot(e, l, color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    fig.tight_layout()  # otherwise the right y-label is slightly clipped
    fig.savefig('loss_acc_{}_{}.png'.format(tag1, tag2))
    plt.close()
#===============================================================================================================================
def eval_model(model, x_train, y_train, x_test, y_test):
    eval_train = model.evaluate(x_train, y_train)
    eval_test = model.evaluate(x_test, y_test)
    
    y_pred = model.predict(x_test)
    y_pred = (y_pred>0.5)
    
    cm = confusion_matrix(y_test, y_pred) 
    
    print('\nPré-Classificador\n')
    print('Acertos - Casos Estáveis: {}, Casos Instáveis: {}\nErros - Falsa Estabilidade: {}, Falsa Instabilidade: {}\n'.format(cm[0][0],cm[1][1],cm[0][1],cm[1][0]))
    print('Precisão de Treino: {}%\nPrecisão de Teste: {}%'.format(eval_train[1]*100,eval_test[1]*100))
    
    return cm[0][1], eval_test[1]*100
#===============================================================================================================================    
def train_test_loop(x_data_norm, y_data, dataset, plot):                              
    print('Executando loop para o conjunto {}'.format(dataset))
    
    x_train, x_test, y_train, y_test = train_test_split(x_data_norm, y_data, 
                                                    test_size=0.3, 
                                                    shuffle = True)
    n_layers = [1,2,3,4]
    acc_desired = [0.92,0.95,0.98]
    architecture = {'1':[[16],[32],[64],[128]],
                    '2':[[16,8],[32,16],[64,32],[128,64]],
                    '3':[[16,8,4],[32,16,8],[64,32,16],[128,64,32]],
                    '4':[[16,8,4,2],[32,16,8,4],[64,32,16,8],[128,64,32,16]]}
    cases = []
    
    for i in range(np.size(n_layers)):
        cases.append(list(product(acc_desired,architecture[str(n_layers[i])])))
        
    print('Iniciando Treinamento com {} casos de {} variáveis'.format(x_train.shape[0], x_train.shape[1]))
    melhores_treinos = []
    melhor_caso = []
    h=0
    for n in range(np.size(n_layers)):
        epocas, acc, arquitetura = [], [], []
        for i,c in enumerate(cases[n]):
            class callback(keras.callbacks.Callback):
                def __init__(self, c, acc_threshold, print_msg):
                    self.acc_threshold = acc_threshold
                    self.print_msg = print_msg
                def on_epoch_end(self, epoch, logs={}):
                    if(logs.get('acc')>self.acc_threshold):
                        if self.print_msg:
                            print('\nAtingiu {}% de precisão na {}ª época - cancelando treino...\n'.format(c[0]*100, 
                                                                                               np.size(model.history.epoch)+1))
                            self.model.stop_training=True
                        else:
                            if self.print_msg:
                                print('\nPrecisão insuficiente - começando nova época\n')
                                print("-"*120)
            #print("Precisão Alvo: {}%\nNº de Camadas: {}\nArquitetura da Rede Neural: {}\n".format(c[0]*100,n+1,c[1]))
            callbacks = callback(c, acc_threshold=c[0], print_msg=True)
            model = build_model(n_layers=n, architecture=c[1], act_func='relu',
                                input_shape=x_train.shape[1], output=1)
            model = compile_train_model(model, x_train, y_train, lr=0.001, 
                                        batch_size=64, epochs=150, verbose=0, callbacks=[callbacks])
            if plot:
                title = "Perda e precisão por época -Precisão Alvo: {}%,Nº de Camadas: {},Arquitetura da Rede Neural: {}".format(c[0]*100,n+1,c[1])
                plot_loss_acc(model,target_acc=c[0],title=title, tag1=h, tag2=dataset)
                
            h=h+1
            
            avaliar_modelo, precisao_teste = eval_model(model, x_train, y_train, x_test, y_test)
            
            epocas.append(np.size(model.history.epoch))
            acc.append(precisao_teste)
            arquitetura.append(c[1])
            
            if avaliar_modelo == 0:
                melhor_caso.append('Melhor Caso: {} falsa estabilidade - Dataset: {}, {} Camadas, Arquitetura {}.\n'.format(avaliar_modelo, dataset, n+1, c[1]))

            print('\nDataset: {} - Arquitetura: {}\n'.format(dataset, c[1]))
            print("-"*70)
        melhores_treinos.append('Dataset: {} - Melhor Treino p/ {} Camada(s): {} Épocas com arquitetura {} e precisão de {}%.\n'.format(dataset,n+1,
                                                                                     np.amin(epocas),
                                                                                     arquitetura[np.argmin(epocas)],
                                                                                     acc[np.argmin(epocas)]))
    return melhores_treinos, melhor_caso


# In[4]:


def train_test_loop_no_threshold(x_data_norm, y_data, dataset, plot, epochs):                              
    print('Executando loop para o conjunto {}'.format(dataset))
    
    x_train, x_test, y_train, y_test = train_test_split(x_data_norm, y_data, 
                                                    test_size=0.3, 
                                                    shuffle = True)
    n_layers = [1,2,3,4]
    acc_desired = [1]
    architecture = {'1':[[16],[32],[64],[128]],
                    '2':[[16,8],[32,16],[64,32],[128,64]],
                    '3':[[16,8,4],[32,16,8],[64,32,16],[128,64,32]],
                    '4':[[16,8,4,2],[32,16,8,4],[64,32,16,8],[128,64,32,16]]}
    cases = []
    
    for i in range(np.size(n_layers)):
        cases.append(list(product(acc_desired,architecture[str(n_layers[i])])))
        
    print('Iniciando Treinamento com {} casos de {} variáveis'.format(x_train.shape[0], x_train.shape[1]))
    melhores_treinos = []
    melhor_caso = []
    h=0
    for n in range(np.size(n_layers)):
        epocas, acc, arquitetura = [], [], []
        for i,c in enumerate(cases[n]):
            
            model = build_model(n_layers=n, architecture=c[1], act_func='relu',
                                input_shape=x_train.shape[1], output=1)
            model = compile_train_model_no_threshold(model, x_train, y_train, lr=0.001, 
                                        batch_size=64, epochs=epochs, verbose=0)
            
            if plot:
                title = "Perda e precisão por época -Precisão Alvo: {}%,Nº de Camadas: {},Arquitetura da Rede Neural: {}".format(c[0]*100,n+1,c[1])
                plot_loss_acc(model,target_acc=c[0],title=title, tag1=h, tag2=dataset)
                
            h=h+1
            
            avaliar_modelo, precisao_teste = eval_model(model, x_train, y_train, x_test, y_test)
            
            epocas.append(np.size(model.history.epoch))
            acc.append(precisao_teste)
            arquitetura.append(c[1])
            
            if avaliar_modelo == 0:
                melhor_caso.append('Melhor Caso: {} falsa estabilidade - Dataset: {}, {} Camadas, Arquitetura {}.\n'.format(avaliar_modelo, dataset, n+1, c[1]))

            print('\nDataset: {} - Arquitetura: {}\n'.format(dataset, c[1]))
            print("-"*70)
        melhores_treinos.append('Dataset: {} - Melhor Treino p/ {} Camada(s): {} Épocas com arquitetura {} e precisão de {}%.\n'.format(dataset,n+1,
                                                                                     np.amin(epocas),
                                                                                     arquitetura[np.argmin(epocas)],
                                                                                     acc[np.argmin(epocas)]))
    return melhores_treinos, melhor_caso

def compile_train_model_no_threshold(model, x_train, y_train, lr, batch_size, epochs, verbose):
    model_copy = model
    
    model_copy.compile(optimizer=SGD(lr=lr),
                      loss='binary_crossentropy',
                      metrics=['accuracy'])
    
    model_copy.fit(x_train, y_train, epochs=epochs, validation_split=0.175, batch_size=batch_size, verbose=verbose)
    return model_copy


# In[5]:


#===============================================================================================================================
#A função 'saida_svm()' tem como saída duas tabelas constrúidas a partir da função 'svm()' para cinco valores de C
def tabelas_svm(subset, y):
    result_treino = list()
    result_teste = list()
    
    C = [10, 1, 0.1, 0.01, 0.001]
    for i in range(4):
        temp=svm(subset, y, C[i], 'linear')
        result_treino.append(temp[0])
        result_teste.append(temp[1])
    
    svm_resultado_teste = pd.DataFrame({'Precisão de Teste (%), C=10': result_teste[0],
                                        'Precisão de Teste (%), C=1': result_teste[1],
                                        'Precisão de Teste (%), C=0.1': result_teste[2],
                                        'Precisão de Teste (%), C=0.01': result_teste[3],
                                        'Precisão de Teste (%), C=0.001': result_teste[4]},
                                       index = index) #Construção da tabela de resultados da SVM para teste

    svm_resultado_treino = pd.DataFrame({'Precisão de Treino (%), C=10': result_treino[0],
                                    'Precisão de Treino (%), C=1': result_treino[0],
                                     'Precisão de Treino (%), C=0.1': result_treino[0],
                                     'Precisão de Treino (%), C=0.01': result_treino[0],
                                     'Precisão de Treino (%), C=0.001': result_treino[0]}, index = index) #Construção da tabela de resultados da SVM para treino
    #p/ resultado percentual
    svm_resultado_treino = svm_resultado_treino.apply(lambda x: x*100)
    svm_resultado_teste = svm_resultado_teste.apply(lambda x: x*100)
    
    return svm_resultado_treino, svm_resultado_teste

def tabelas_svm_rfecv(teste, treino):

    
    svm_resultado_teste = pd.DataFrame({'Precisão de Teste (%), C=10': result_teste[0],
                                        'Precisão de Teste (%), C=1': result_teste[1],
                                        'Precisão de Teste (%), C=0.1': result_teste[2],
                                        'Precisão de Teste (%), C=0.01': result_teste[3],
                                        'Precisão de Teste (%), C=0.001': result_teste[4]},
                                       index = index) #Construção da tabela de resultados da SVM para teste

    svm_resultado_treino = pd.DataFrame({'Precisão de Treino (%), C=10': result_treino[0],
                                    'Precisão de Treino (%), C=1': result_treino[1],
                                     'Precisão de Treino (%), C=0.1': result_treino[2],
                                     'Precisão de Treino (%), C=0.01': result_treino[3],
                                     'Precisão de Treino (%), C=0.001': result_treino[4]}, index = index) #Construção da tabela de resultados da SVM para treino
    #p/ resultado percentual
    svm_resultado_treino = svm_resultado_treino.apply(lambda x: x*100)
    svm_resultado_teste = svm_resultado_teste.apply(lambda x: x*100)
    
    return svm_resultado_treino, svm_resultado_teste


# In[6]:


def SVM_optimal(conjuntos, y_data, cv):
    result_test = pd.DataFrame()
    result_train = pd.DataFrame()
    for conjunto in conjuntos:
        x_data = conjuntos[str(conjunto)]
        estimador = SVC(kernel='linear')
        C_range = np.logspace(-10, 1, 12)
        
        pontuacao = list()
        pontuacao_std = list()

        for C in C_range:
            estimador.C = C
            c_pontuacao = cross_val_score(estimador, x_data, y_data, cv=cv, n_jobs=-1)
            pontuacao.append(np.mean(c_pontuacao))
            pontuacao_std.append(np.std(c_pontuacao))
        
        acc_treino, acc_teste = SVM_RFECV(x_data, y_data, C=C_range[np.array(pontuacao).argmax()], cv=cv)
        result_test.append(pd.DataFrame({'Precisão de Teste (%)': acc_teste, 'C': C_range[np.array(pontuacao).argmax()]/100},
                                         index = index))
        result_train.append(pd.DataFrame({'Precisão de Treino (%)': acc_treino, 'C': C_range[np.array(pontuacao).argmax()]/100}, 
                                         index = index))
    
    result_train = result_train.apply(lambda x: x*100)
    result_test = result_test.apply(lambda x: x*100)
    return result_train, result_test      


# In[ ]:




