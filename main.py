from gui import get_examiner_window
import sys

def main(args):
    window = get_examiner_window()
    window.mainloop()


if __name__ == '__main__':
    main(sys.argv)

