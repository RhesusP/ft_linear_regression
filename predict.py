# Chargement des paramètres theta depuis le fichier
file = open(".theta", "r")
a = float(file.readline())  # theta_1
b = float(file.readline())  # theta_0
file.close()

# Demander à l'utilisateur d'entrer le kilométrage
mileage = input("Enter the mileage: ")
mileage = float(mileage)
if mileage < 0:
    print("The mileage cannot be negative")
    exit()

# f(x) = a * x + b
predicted_price = (a * mileage) + b

# Affichage du prix prédit
print(f"The predicted price for {mileage} km is: {predicted_price}")
