import pandas as pd
import matplotlib.pyplot as plt

ba =pd.read_csv("Day_GOLD_20231200.csv")

ba['Mean'] = (ba['High'] + ba['Low']) / 2




# Plot "High", "Low", and "Mean" columns
plt.figure(figsize=(10, 6))
plt.plot(ba.index, ba['High'], label='High', color='blue')
plt.plot(ba.index, ba['Low'], label='Low', color='green')
plt.plot(ba.index, ba['Mean'], label='Mean', color='red')
plt.title('High, Low, and Mean of Futures Data')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()
plt.grid()
plt.tight_layout()

# Save the plot as an image
plt.savefig(f'futures_data.png')
plt.close()

# Save the processed data to an Excel file
excel_path = f'Futures_Market_Data.xlsx'
with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
    ba.to_excel(writer, sheet_name='Raw Data')
    print(f"Data saved to Excel at {excel_path}")
