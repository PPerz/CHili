import cv2
import numpy as np
import datetime

def process_leaf_image(image_path="Chili.jpg"):
    # Wczytaj obraz
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Zastosuj filtr Gaussa i progowanie
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresh = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Znajdź kontury
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Utwórz pusty obraz o takich samych wymiarach
    contour_image = np.zeros_like(image)
    
    # Narysuj kontury na pustym obrazie (niebieski kolor, grubość 2 px)
    cv2.drawContours(contour_image, contours, -1, (255, 0, 0), 2)
    
    # Generowanie nazwy pliku wyjściowego z datą
    current_date = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_path = f"Kontury_{current_date}.jpg"
    
    # Zapisz wynikowy obraz do pliku JPG
    cv2.imwrite(output_path, contour_image)
    print(f"Obraz konturów zapisany jako {output_path}")
    
    # Pokaż obraz (opcjonalne)
    cv2.imshow("Kontury Liści", contour_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Przykład użycia
process_leaf_image()