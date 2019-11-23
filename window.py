import sys
import tkinter as tk
from tkinter import *
from tkinter import ttk

def button_clicked(event):
    val = v1.get()
    lv = tk.Label(text=val)
    lv.pack_forget()
    lv.pack(side=tk.BOTTOM)

def cb_selected(event):
    print('v1 = %s' % v1.get())

if __name__ == "__main__":
    root = tk.Tk()
    root.title(u"select kigo")
    width = 800
    height = 600
    root.geometry("{0}x{1}".format(width,height))

    v1 = StringVar()
    cb = ttk.Combobox(textvariable=v1)
    cb.bind('<<ComboboxSelected>>', cb_selected)

    cb['values']=("桜","海","紅葉","雪")
    cb.set("桜")
    cb.pack()

    Button = tk.Button(text="OK", width=10)
    Button.bind("<Button-1>",button_clicked)
    Button.pack()

    root.mainloop()