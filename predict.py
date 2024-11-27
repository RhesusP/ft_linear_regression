# Load thetas from .theta file
file = open(".theta", "r")
a = float(file.readline())  # theta_1
b = float(file.readline())  # theta_0
file.close()

# Mileage input
mileage = input("Enter the mileage: ")
mileage = float(mileage)
if mileage < 0:
    print("The mileage cannot be negative")
    exit()

# f(x) = a * x + b
predicted_price = (a * mileage) + b

print(f"The predicted price for {mileage} km is: {predicted_price}")
