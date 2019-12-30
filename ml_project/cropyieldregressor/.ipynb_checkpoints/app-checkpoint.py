{
 "cells": [
  {
   "cell_type": "code",
   "execution_count": 2,
   "metadata": {},
   "outputs": [
    {
     "ename": "SyntaxError",
     "evalue": "invalid syntax (<ipython-input-2-dd4694aaf67a>, line 7)",
     "output_type": "error",
     "traceback": [
      "\u001b[1;36m  File \u001b[1;32m\"<ipython-input-2-dd4694aaf67a>\"\u001b[1;36m, line \u001b[1;32m7\u001b[0m\n\u001b[1;33m    if cropyieldregressor == '_main_'\u001b[0m\n\u001b[1;37m                                     ^\u001b[0m\n\u001b[1;31mSyntaxError\u001b[0m\u001b[1;31m:\u001b[0m invalid syntax\n"
     ]
    }
   ],
   "source": [
    "from flask import Flask, render_template\n",
    "\n",
    "app = Flask(cropyieldregressor)\n",
    "@app.route('/')\n",
    "def index():\n",
    "    return render_template('crop_yield_app.html')\n",
    "if cropyieldregressor == '_main_'\n",
    "app.run()"
   ]
  },
  {
   "cell_type": "code",
   "execution_count": null,
   "metadata": {},
   "outputs": [],
   "source": []
  }
 ],
 "metadata": {
  "kernelspec": {
   "display_name": "Python 3",
   "language": "python",
   "name": "python3"
  },
  "language_info": {
   "codemirror_mode": {
    "name": "ipython",
    "version": 3
   },
   "file_extension": ".py",
   "mimetype": "text/x-python",
   "name": "python",
   "nbconvert_exporter": "python",
   "pygments_lexer": "ipython3",
   "version": "3.7.5"
  }
 },
 "nbformat": 4,
 "nbformat_minor": 2
}
