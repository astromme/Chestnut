#!/usr/bin/env python

from PyQt4 import QtGui, QtCore, QtOpenGL
from PyQt4.QtOpenGL import *

from pygments import highlight
from pygments.lexers import *
from pygments.formatter import Formatter

import sys, re, time, traceback

import chestnut.parser

# keep track of our currently loaded modules so we know which ones to reload
initialized_modules = sys.modules.keys()

def hex2QColor(c):
    r=int(c[0:2],16)
    g=int(c[2:4],16)
    b=int(c[4:6],16)
    return QtGui.QColor(r,g,b)

class QFormatter(Formatter):

    def __init__(self):
        Formatter.__init__(self)
        self.data=[]

        # Create a dictionary of text styles, indexed
        # by pygments token names, containing QTextCharFormat
        # instances according to pygments' description
        # of each style

        self.styles={}
        for token, style in self.style:
            qtf=QtGui.QTextCharFormat()

            if style['color']:
                qtf.setForeground(hex2QColor(style['color']))
            if style['bgcolor']:
                qtf.setBackground(hex2QColor(style['bgcolor']))
            if style['bold']:
                qtf.setFontWeight(QtGui.QFont.Bold)
            if style['italic']:
                qtf.setFontItalic(True)
            if style['underline']:
                qtf.setFontUnderline(True)
            self.styles[str(token)]=qtf

    def format(self, tokensource, outfile):
        global styles
        # We ignore outfile, keep output in a buffer
        self.data=[]

        # Just store a list of styles, one for each character
        # in the input. Obviously a smarter thing with
        # offsets and lengths is a good idea!

        for ttype, value in tokensource:
            l=len(value)
            t=str(ttype)
            self.data.extend([self.styles[t],]*l)


class Highlighter(QtGui.QSyntaxHighlighter):

    def __init__(self, parent, mode):
        QtGui.QSyntaxHighlighter.__init__(self, parent)
        self.tstamp=time.time()

        # Keep the formatter and lexer, initializing them 
        # may be costly.
        self.formatter=QFormatter()
        self.lexer=get_lexer_by_name(mode)

    def highlightBlock(self, text):
        """Takes a block, applies format to the document.
        according to what's in it.
        """

        # I need to know where in the document we are,
        # because our formatting info is global to
        # the document
        cb = self.currentBlock()
        p = cb.position()

        # The \n is not really needed, but sometimes  
        # you are in an empty last block, so your position is
        # **after** the end of the document.
        text=unicode(self.document().toPlainText())+'\n'

        # Yes, re-highlight the whole document.
        # There **must** be some optimizacion possibilities
        # but it seems fast enough.
        highlight(text,self.lexer,self.formatter)

        # Just apply the formatting to this block.
        # For titles, it may be necessary to backtrack
        # and format a couple of blocks **earlier**.
        for i in range(len(unicode(text))):
            try:
                self.setFormat(i,1,self.formatter.data[p+i])
            except IndexError:
                pass

        # I may need to do something about this being called
        # too quickly.
        self.tstamp=time.time()

from PyQt4.Qt import QFrame
from PyQt4.Qt import QHBoxLayout
from PyQt4.Qt import QPainter
from PyQt4.Qt import QPlainTextEdit
from PyQt4.Qt import QRect
from PyQt4.Qt import QTextEdit
from PyQt4.Qt import QTextFormat
from PyQt4.Qt import QVariant
from PyQt4.Qt import QWidget
from PyQt4.Qt import Qt

class LNTextEdit(QFrame):
    class NumberBar(QWidget):

        def __init__(self, edit):
            QWidget.__init__(self, edit)

            self.edit = edit
            self.adjustWidth(1)

        def paintEvent(self, event):
            self.edit.numberbarPaint(self, event)
            QWidget.paintEvent(self, event)

        def adjustWidth(self, count):
            width = self.fontMetrics().width(unicode(count))
            if self.width() != width:
                self.setFixedWidth(width)

        def updateContents(self, rect, scroll):
            if scroll:
                self.scroll(0, scroll)
            else:
                # It would be nice to do
                # self.update(0, rect.y(), self.width(), rect.height())
                # But we can't because it will not remove the bold on the
                # current line if word wrap is enabled and a new block is
                # selected.
                self.update()


    class PlainTextEdit(QPlainTextEdit):
        def __init__(self, *args):
            QPlainTextEdit.__init__(self, *args)
            #self.setFrameStyle(QFrame.NoFrame)

            self.setFrameStyle(QFrame.NoFrame)
            self.highlight()
            #self.setLineWrapMode(QPlainTextEdit.NoWrap)

            self.cursorPositionChanged.connect(self.highlight)

        def highlight(self):
            hi_selection = QTextEdit.ExtraSelection()

            hi_selection.format.setBackground(self.palette().alternateBase())
            hi_selection.format.setProperty(QTextFormat.FullWidthSelection, QVariant(True))
            hi_selection.cursor = self.textCursor()
            hi_selection.cursor.clearSelection()

            self.setExtraSelections([hi_selection])

        def numberbarPaint(self, number_bar, event):
            font_metrics = self.fontMetrics()
            current_line = self.document().findBlock(self.textCursor().position()).blockNumber() + 1

            block = self.firstVisibleBlock()
            line_count = block.blockNumber()
            painter = QPainter(number_bar)
            painter.fillRect(event.rect(), self.palette().base())

            # Iterate over all visible text blocks in the document.
            while block.isValid():
                line_count += 1
                block_top = self.blockBoundingGeometry(block).translated(self.contentOffset()).top()

                # Check if the position of the block is out side of the visible
                # area.
                if not block.isVisible() or block_top >= event.rect().bottom():
                    break

                # We want the line number for the selected line to be bold.
                if line_count == current_line:
                    font = painter.font()
                    font.setBold(True)
                    painter.setFont(font)
                else:
                    font = painter.font()
                    font.setBold(False)
                    painter.setFont(font)

                # Draw the line number right justified at the position of the line.
                paint_rect = QRect(0, block_top, number_bar.width(), font_metrics.height())
                painter.drawText(paint_rect, Qt.AlignRight, unicode(line_count))

                block = block.next()

            painter.end()

    def __init__(self, *args):
        QFrame.__init__(self, *args)

        self.setFrameStyle(QFrame.StyledPanel | QFrame.Sunken)

        self.edit = self.PlainTextEdit()
        self.number_bar = self.NumberBar(self.edit)

        hbox = QHBoxLayout(self)
        hbox.setSpacing(0)
        hbox.setMargin(0)
        hbox.addWidget(self.number_bar)
        hbox.addWidget(self.edit)

        self.edit.blockCountChanged.connect(self.number_bar.adjustWidth)
        self.edit.updateRequest.connect(self.number_bar.updateContents)

    def toPlainText(self):
        return unicode(self.edit.toPlainText())

    def setPlainText(self, text):
        self.edit.setPlainText(text)

    def isModified(self):
        return self.edit.document().isModified()

    def setModified(self, modified):
        self.edit.document().setModified(modified)

    def setLineWrapMode(self, mode):
        self.edit.setLineWrapMode(mode)


class InternalsViewer(QtGui.QMainWindow):
    def __init__(self):
        QtGui.QMainWindow.__init__(self)

        self.setWindowTitle("Chestnut Internals Viewer")

        self.widget = QtGui.QWidget()
        self.layout = QtGui.QHBoxLayout()
        self.splitter = QtGui.QSplitter()

        ## Toolbars
        self.toolbar = self.addToolBar('Actions')

        self.reloadParserAction = QtGui.QAction("Reload Parser", self)
        self.reloadParserAction.setShortcut(QtGui.QKeySequence('Ctrl+Shift+T'))
        self.reloadParserAction.triggered.connect(self.reload_parser)

        self.reloadAction = QtGui.QAction("Reload", self)
        self.reloadAction.setShortcut(QtGui.QKeySequence('Ctrl+Shift+R'))
        self.reloadAction.triggered.connect(self.reload_action)

        self.runAction = QtGui.QAction("Run", self)
        self.runAction.setShortcut(QtGui.QKeySequence('Ctrl+R'))
        self.runAction.triggered.connect(self.update_code)

        self.toolbar.addAction(self.reloadParserAction)
        self.toolbar.addAction(self.reloadAction)
        self.toolbar.addAction(self.runAction)

        ## Code Windows
        font = QtGui.QFont("Courier New");
        font.setPixelSize(14);

        self.chestnutWindow = LNTextEdit()#QtGui.QPlainTextEdit()
        self.chestnutWindow.setLineWrapMode(QtGui.QPlainTextEdit.NoWrap)
        self.chestnutWindow.setFont(font)
        self.source_hl=Highlighter(self.chestnutWindow.edit.document(), "c++")

        self.astWindow = LNTextEdit()#QtGui.QPlainTextEdit()
        self.astWindow.setLineWrapMode(QtGui.QPlainTextEdit.NoWrap)
        self.astWindow.setFont(font)

        self.compiledWindow = LNTextEdit()#QtGui.QPlainTextEdit()
        self.compiledWindow.setLineWrapMode(QtGui.QPlainTextEdit.NoWrap)
        self.compiledWindow.setFont(font)
        self.dest_hl=Highlighter(self.compiledWindow.edit.document(), "c++")

        #self.layout.addWidget(self.reloadButton)
        #self.layout.addWidget(self.runButton)

        self.splitter.addWidget(self.chestnutWindow)
        self.splitter.addWidget(self.astWindow)
        self.splitter.addWidget(self.compiledWindow)

        self.layout.addWidget(self.splitter)

        self.widget.setLayout(self.layout)
        self.setCentralWidget(self.widget)

        self.timestep = 0
        self.time = QtCore.QTime()
        self.time.start()
        self.last_time = 0

        self.chestnutWindow.setPlainText("""\
IntArray1d numbers[10] = read("hello");

foreach num in numbers
    num = 4;
    num += 4;
    num += num.east;
end

//sequential Int some_function(Real var, Int var2) {
//  print("Var: %s, Var2: %s", var, var2);
//  print("blah blah blah");
//}
//
//some_function(4.2, 10);
""")

    def reload_action(self):
        self.reload_modules()
        self.update_code()

    def reload_parser(self):
        print 'reloading parser'
        del(sys.modules['chestnut.parser'])
        self.update_code()

    def reload_modules(self):
        # remove modules so they can be reloaded
        print "removing modules"
        for m in [x for x in sys.modules.keys() if x not in initialized_modules]:
            del(sys.modules[m])

    def update_code(self):
        print "lodaing chestnut"
        from chestnut.backends.walnut import WalnutBackend
        from chestnut.exceptions import CompilerException
        from chestnut.parser import parse, FullFirstMatchException
        from chestnut.symboltable import SymbolTable

        print "parsing"

        # Time to get an AST, but that might fail and we want nice lines + context information
        code = str(self.chestnutWindow.toPlainText())
        try:
            ast = parse(code)
        except FullFirstMatchException, e:
            code_lines = code.split('\n')
            lineno = e.kargs['lineno']

            def valid_line(lineno):
                return lineno > 0 and lineno < len(code_lines)

            import textwrap
            error = ""
            error += 'Error:\n'
            error += '\n'.join(map(lambda s: '    ' + s, textwrap.wrap(str(e), 76))) + '\n'
            error += 'Context:\n'
            for offset in [-2, -1, 0, 1, 2]:
                if valid_line(lineno+offset):
                    if offset == 0: error += '  ->'
                    else: error += '    '
                    error += '%s: %s\n' % (lineno+offset, code_lines[lineno+offset - 1])
            self.astWindow.setPlainText(error)
            return

        self.astWindow.setPlainText(str(ast))

        print "compiling"

        # Time to compile to C++ code, but that might fail and we want nice error messages
        try:
            backend = WalnutBackend()
            thrust_code = backend.compile(ast)
            self.compiledWindow.setPlainText(thrust_code)
        except CompilerException as e:
            self.compiledWindow.setPlainText(str(e))
        except Exception as e:
            self.compiledWindow.setPlainText(traceback.format_exc(e))


if __name__ == '__main__':
    app = QtGui.QApplication(['Chestnut Internals Viewer'])
    window = InternalsViewer()
    window.show()

    app.exec_()

