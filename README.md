# Tracking und Computer Vision für Augmented Reality

Enhält die Folien und den Code für die Vorlesung vom 13.06.2017 (Datei slides.pdf).

Um den Code zum Laufen zu Bringen muss Python inkl. einiger Bibliotheken (numpy, OpenCV, pyglet) installiert sein. Für die Notebooks wird Jupyter benötigt. Am Einfachsten ist eine Installation von [Anaconda mit Python 2.7](https://www.continuum.io/downloads). Zusätzlich muss wahrscheinlich noch installiert werden:

 * OpenCV über `conda install -c menpo opencv=2.4.11`
 * pyglet über `pip install pyglet` oder `conda install -c conda-forge pyglet=1.2.4`

Der ARTag Marker kann selbst ausgedruckt werden (Datei marker.png).

Der Importer für 3D-Objekte im OBJ Format ist nicht von mir, sondern von [hier geklaut](https://github.com/greenmoss/pyglet_obj_test/blob/master/importers.py). Das 3D-Objekt (Affe) ist von [Blender](https://www.blender.org/).