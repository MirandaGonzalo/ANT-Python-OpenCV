from tutorial_basico import *
from tutorial_avanzado import *
import time

print "Bienvenidos al tutorial de Python OpenCV"
tutorial = True

while tutorial:
    print "Elija que hacer"
    print "Opciones:"
    print ""
    print "|"
    print "|"
    print "\ - - - > Salir (0)"
    print "|"
    print "\ - - - > Tutorial Basico (1)"
    print "|"
    print "\ - - - > Tutorial Avanzado (2)"
    print ""
    
    eleccion = input()
    
    if eleccion == 0:
        exit()
    if eleccion == 1:
        print "Bienvenido al tutorial Basico de Python OpenCV"
        print "Selecciona la letra con la que se va a cerrar la ventana"
        aux = raw_input()
        
        infinito = True
        while infinito:
            print "Elija que quiere hacer"
            print "Opciones:"
            print ""
            print "|"
            print "|"
            print "\ - - - > Salir (0)"
            print "|"
            print "\ - - - > Mostrar imagen sin filtro (1)"
            print "|"
            print "\ - - - > Mostrar imagen con filtro gris (2)"
            print "|"
            print "\ - - - > Guardar la imagen con filtro gris (3)"
            print "|"
            print "\ - - - > Grabar un video desde la camara (4)"
            print "|"
            print "\ - - - > Ver Colores (5)"
            print ""

            eleccion2 = input()
            print ""

            if eleccion2 == 0:
                break
            if eleccion2 == 1:
                image_without_cvtcolor()
                time.sleep (1)
            if eleccion2 == 2:
                grey()
                time.sleep (1)
            if eleccion2 == 3:
                if save_gray_image():
                    print "Guardado"
                    time.sleep (1)
                else:
                    print "No se Guardo"
                    time.sleep (1)
            if eleccion2 == 4:
                camara()
                time.sleep (1)
            if eleccion2 == 5:
                ver_colores()
                time.sleep(1)