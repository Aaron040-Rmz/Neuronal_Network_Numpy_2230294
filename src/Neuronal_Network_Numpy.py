import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_gaussian_quantiles

def train_neural_network():
  """
  Función principal para entrenar la red neuronal y visualizar los resultados.
  """
  # Creemos datasets desde cero - Para un ejemplo de clasificacion
  N = 1000
  gaussian_quantiles = make_gaussian_quantiles(
    mean=None,
    cov=0.1,
    n_samples=N,
    n_features=2,
    n_classes=2,
    shuffle=True,
    random_state=None)

  X, Y = gaussian_quantiles
  Y = Y[:,np.newaxis]

  print(X.shape)
  print(Y.shape)
  plt.figure()
  plt.scatter(X[:,0], X[:,1],c=Y, s=40, cmap=plt.cm.Spectral)
  plt.show()
  
  # Funciones de activacion

  def sigmoid(x, derivate=False):
      """
      Función de activación sigmoide para introducir no linealidad.
      Si derivate=True, devuelve la derivada de la función.
      """

      if derivate:
          return np.exp(-x)/(np.exp(-x)+1)**2 # Derivada de la sigmoide
      else:
          return 1/(1+np.exp(-x))# Valor de la sigmoide

  def relu(x, derivate=False):
      """
      Función de activación ReLU, para capas ocultas.
      Si derivate=True, devuelve la derivada de ReLU.
      """

      if derivate:
          x[x<=0]=0 # Derivada es 0 para valores negativos
          x[x>0]=1 # Derivada es 1 para valores positivos
          return x
      else:
          return np.maximum(0,x)# ReLU devuelve el mayor entre 0 y x


  # Funcion de perdida
  def mse(y, y_hat, derivate=False):
      """
      Calcula el Error Cuadrático Medio (MSE) como función de pérdida.
      Si derivate=True, devuelve la derivada del MSE.
      """
      
      if derivate:
          return (y_hat-y)# Derivada del MSE respecto a la salida
      else:
          return np.mean((y_hat-y)**2) # Cálculo del MSE


  # Estructura de la red: asignar pesos y biases

  def initialize_parameters_deep(layers_dims):
      """
      Inicializa los pesos y sesgos de cada capa de la red neuronal.
      layers_dims: lista con el número de neuronas por capa.
      """
      parameters = {}
      L = len(layers_dims)# Cantidad de capas (incluyendo entrada y salida)
      for l in range(0,L-1):
          parameters['W'+str(l+1)] = (np.random.rand(
                          layers_dims[l], layers_dims[l+1])*2)-1
          parameters['b'+str(l+1)] = (np.random.rand(1,
                          layers_dims[l+1])*2)-1
      return parameters


  def train(x_data, learning_rate, params, training=True):
      """
      Realiza la propagación hacia adelante y el entrenamiento si training=True.
      """

      params['A0']=x_data# Entrada a la red


      # Capa 1
      params['Z1']=np.matmul(params['A0'], params['W1'])+params['b1']
      params['A1']=relu(params['Z1'])

      # Capa 2
      params['Z2']=np.matmul(params['A1'], params['W2'])+params['b2']
      params['A2']=relu(params['Z2'])

      # Capa 3 (salida)
      params['Z3']=np.matmul(params['A2'], params['W3'])+params['b3']
      params['A3']=sigmoid(params['Z3'])

      output = params['A3']

      if training:
          # Propagación hacia atrás para ajustar pesos y sesgos
          params['dZ3'] = mse(Y,output,True)*sigmoid(params['A3'],True)
          params['dW3']=np.matmul(params['A2'].T, params['dZ3'])

          
          params['dZ2']=np.matmul(params['dZ3'], params['W3'].T)*relu(params['A2'],True)
          params['dW2']=np.matmul(params['A1'].T, params['dZ2'])

       
          params['dZ1']=np.matmul(params['dZ2'], params['W2'].T)*relu(params['A1'], True)
          params['dW1']=np.matmul(params['A0'].T, params['dZ1'])

          # Actualización de parámetros (gradiente descendente)
          params['W3']=params['W3']-params['dW3']*learning_rate
          params['W2']=params['W2']-params['dW2']*learning_rate
          params['W1']=params['W1']-params['dW1']*learning_rate

          params['b3']=params['b3']-(np.mean(params['dW3'], axis=0, keepdims=True))*learning_rate
          params['b2']=params['b2']-(np.mean(params['dW2'], axis=0, keepdims=True))*learning_rate
          params['b1']=params['b1']-(np.mean(params['dW1'], axis=0, keepdims=True))*learning_rate

      return output

   # Preparar los datos de entrenamiento
  layers_dims = [2, 6, 10, 1]  # Estructura de la red: 2 entradas, 2 capas ocultas, 1 salida
  params = initialize_parameters_deep(layers_dims)
  errors = [] # Lista para guardar el error en cada iteración

  # Entrenamiento de la red neuronal
  for _ in range(50000):
      output = train(X, 0.001, params)
      if _%50==0:  # Cada 50 iteraciones muestra el error
          print(mse(Y,output)) # Imprime el error actual
          errors.append(mse(Y,output)) # Guarda el error

  plt.plot(errors)
  # Crear y clasificar nuevos datos para probar la red entrenada
  data_test_x=(np.random.rand(1000,2)*2)-1
  data_test_y = train(data_test_x,0.0001,params,training=False)

  y= np.where(data_test_y>0.5,1,0)
  plt.figure()
  plt.scatter(data_test_x[:,0], data_test_x[:,1],c=y, s=40, cmap=plt.cm.Spectral)
  plt.show()