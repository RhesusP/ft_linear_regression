def main():
    # Load thetas from .theta file
    try:
        file = open(".theta", "r")
    except Exception as e:
        print("Error: Unable to read thetas (", e, ")")
        exit(1)
    try:
        a = float(file.readline())  # theta_1
        b = float(file.readline())  # theta_0
    except Exception as e:
        print("Error: Thetas are not valid (", e, ")")
        exit(1)
    file.close()

    # Mileage input
    mileage = input("Enter the mileage: ")
    try:
        mileage = float(mileage)
    except Exception as e:
        print("Error: The mileage must be a number (", e, ")")
        exit(1)
    if mileage < 0:
        print("The mileage cannot be negative")
        exit(1)

    # f(x) = a * x + b
    predicted_price = (a * mileage) + b
    if predicted_price < 0:
        predicted_price = 0

    print(f"The predicted price for {mileage} km is: {round(predicted_price, 2)}")


if __name__ == "__main__":
    main()
