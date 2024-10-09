# BIBLIOTECAS

import os
import cv2  #pip install opencv-contrib-python
import numpy as np

def captura(largura, altura):
    classificador = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    classificador_olho = cv2.CascadeClassifier('haarcascade_eye.xml')
    
    # camera
    camera = cv2.VideoCapture(0)
    
    # amostras do usuário
    amostra = 1
    n_amostras = 25
    
    # recebe o id do usuário
    id = input('Digite o ID do usuário: ')
    
    # mensagem indicando a captura de imagem
    print('Capturando as imagens...')
    
    # loop
    while True:
        conectado, imagem = camera.read()
        imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)
        print(np.average(imagem_cinza))
        faces_detectadas = classificador.detectMultiScale(imagem_cinza, scaleFactor=1.5, minSize=(150,150))
        
        # identifica a geometria das faces
        for (x, y, l, a) in faces_detectadas:
            cv2.rectangle(imagem, (x, y), (x+l, y+a), (0, 0,255),2)
            regiao = imagem[y: y+a, x: x+l]
            regiao_cinza_olho = cv2.cvtColor(regiao, cv2.COLOR_BGR2GRAY)
            olhos_detectados = classificador_olho.detectMultiScale(regiao_cinza_olho)
        
            # identificar a geometria dos olhos
            for(ox, oy, ol, oa) in olhos_detectados:
                cv2.rectangle(regiao, (ox, oy), (ox + ol, oy + oa), (0, 255,0),2)
                
            if np.average(imagem_cinza) > 110 and amostra <= n_amostras:
                imagem_face = cv2.resize(imagem_cinza[y:y+a, x:x + l], (largura, altura))
                cv2.imwrite(f'fotos/pessoa.{str(id)}.{str(amostra)}.jpg', imagem_face)
                print(f'[foto] {str(amostra)} de nome capturada com sucesso.')
                amostra += 1
        cv2.imshow('Detectar faces', imagem)
        cv2.waitKey(1)
        
        if(amostra >= n_amostras + 1):
            print('Faces capturadas com sucesso.')
            break
        elif cv2.waitKey(1) == ord('q'):
            print('Câmera encerrada.')
            break
    
    # encerra a captura
    camera.release()
    cv2.destroyAllWindows()
    # fim da função
    
def get_imagem_com_id():
    caminhos = [os.path.join('fotos', f) for f in os.listdir('fotos')]
    faces = []
    ids = []
    
    for caminho_imagem in caminhos:
        imagem_face = cv2.cvtColor(cv2.imread(caminho_imagem), cv2.COLOR_BGR2GRAY)
        id = int(os.path.split(caminho_imagem)[-1].split('.')[1])
        ids.append(id)
        faces.append(imagem_face)
    
    return np.array(ids), faces


def treinamento():
    # cria os elementos de reconhecimento necessários
    eigenface = cv2.face.EigenFaceRecognizer_create()
    fisherface = cv2.face.FisherFaceRecognizer_create()
    lbph = cv2.face.LBPHFaceRecognizer_create()
    
    ids, faces = get_imagem_com_id()
    
    # treinando o algoritmo do programa
    print('Treinando...')
    eigenface.train(faces, ids)
    eigenface.write('classificadorEigen.yml')
    fisherface.train(faces, ids)
    fisherface.write('classificadorFisher.yml')
    lbph.train(faces, ids)
    lbph.write('classificadorLBPH.yml')
    
    # finaliza treinamento
    print('Treinamento finalizado com sucesso!')
    
def reconhecedor_eigenfaces(largura, altura):
    detector_faces = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    reconhecedor = cv2.face.EigenFaceRecognizer_create()  # Correção do nome da função
    reconhecedor.read("classificadorEigen.yml")
    fonte = cv2.FONT_HERSHEY_COMPLEX_SMALL

    camera = cv2.VideoCapture(0)

    while True:
        conectado, imagem = camera.read()
        imagem_cinza = cv2.cvtColor(imagem, cv2.COLOR_BGR2GRAY)  # Correção da cor
        faces_detectadas = detector_faces.detectMultiScale(imagem_cinza, scaleFactor=1.5, minSize=(30, 30))  # Correção de digitação

        for (x, y, l, a) in faces_detectadas:
            imagem_face = cv2.resize(imagem_cinza[y:y + a, x:x + l], (largura, altura))  # Correção de coordenadas
            cv2.rectangle(imagem, (x, y), (x + l, y + a), (0, 0, 255), 2)
            id, confianca = reconhecedor.predict(imagem_face)
            cv2.putText(imagem, str(id), (x, y + (a + 30)), fonte, 2, (0, 0, 255))

        cv2.imshow("Reconhecer faces", imagem)
        if cv2.waitKey(1) == ord("q"):
            break

    camera.release()
    cv2.destroyAllWindows()
# codigo do eingengral
        
     
#NOTE: programa principal
if __name__ == '__main__':
    # definir tamanho da câmera
    largura = 220
    altura = 220
    
    while True:
        # menu
        print(f'{4* '-'} SIATEMA DE RECONHECIMNTO FACIAL {4* '-'}')
        print('0 - Sair do programa.')
        print('1 - Capturar imagem do usuário.')
        print('2 - Treinar sistema.')
        print('3 - Reconhecer faces.')
        print(f'{41* '-'}')
        opcao = input('Selecione a opção: ')
        
        match opcao:
            case '0':
                print('Programa encerrado.')
                break
            case '1':
                captura(largura, altura)
                continue 
            case '2':
                treinamento()
                continue
            case '3':
                reconhecedor_eigenfaces(largura, altura)
                continue
            case _:
                print('Opção inválida!')
                continue