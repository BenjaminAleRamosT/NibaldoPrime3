# NibaldoPrime

NOTA: 6.0,

al final se realizo un cambio en pos de que funcionara, se añadieron los pesos del decoder a la red final. No es realmente lo que pide el profe, pero se logra llegar a 100 fscore de manera estable.

Esto se logra comentando las lineas siguientes:

![image](https://github.com/BenjaminAleRamosT/NibaldoPrime3/assets/81858128/3cddcf03-2f1b-474f-bc2d-8cb138c13f40)



Va medio del culeque derrepente(los fscores), no se si es el prep o k
la mayoria de las veces tira fscores de 100 y en algunos casos puntuales llega a bajar a 89
Ejemplo, cada columna corresponde a los fscores de una corrida, la ultima fila corresponde a los Fscores promedios

![image](https://github.com/BenjaminAleRamosT/NibaldoPrime3/assets/81858128/073c3b4e-51f8-4796-9ab0-92999bcddbc3)

La curva de costos se ven como poto de wawa

![image](https://github.com/BenjaminAleRamosT/NibaldoPrime3/assets/81858128/a822d92c-7484-4e06-a650-694b03860581)

en el test la funcion de activacion esta hardcodeada, no se si cargar la config y pasarle el param de ahi

![image](https://github.com/BenjaminAleRamosT/NibaldoPrime3/assets/81858128/40bfde8c-4f91-438e-87d6-e62a782ac6ec)

El archivo multipleruns esta hecho para correr el train y el test 5 veces, asegurese de correr el prep primero por su cuenta ( Se utiliza un paquete llamado tqdm , que es solo para ver cuantos segundos demoran cada corrida)

El archivo trn1.py es una version alternativa de trn donde la variable 't' del algoritmo adam corresponde a las iteraciones totales, en trn.py correponde a las iteraciones del minibatch unicamente.

Al realizar el update de pesos utilice una unica funcion, es por eso que en la softmax se hace un poco un truquito de generar una lista de un unico elemento para los pesos, el esqueleto del profe la tenia separada en dos funciones pero la matematica es la misma, se puede cambiar si gusta.

![image](https://github.com/BenjaminAleRamosT/NibaldoPrime3/assets/81858128/176ae5e0-44cc-43c1-9200-493e1e5f74fa)

