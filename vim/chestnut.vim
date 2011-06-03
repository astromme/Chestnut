" Vim syntax file
" Language:	Chestnut
" Maintainer:	Andrew Stromme <astromme@chatonka.com>
" URL:		http://
" Last Change:	2011 Jun 3

if !exists("main_syntax")
  if version < 600
    syntax clear
  elseif exists("b:current_syntax")
    finish
  endif
  let main_syntax = 'chestnut'
endif

syn case match

syn keyword chestnutCommentTodo      TODO FIXME XXX TBD contained
syn match   chestnutLineComment      "\/\/.*" contains=@Spell,javaScriptCommentTodo
syn match   chestnutCommentSkip      "^[ \t]*\*\($\|[ \t]\+\)"
syn region  chestnutComment	       start="/\*"  end="\*/" contains=@Spell,javaScriptCommentTodo
syn match   chestnutSpecial	       "\\\d\d\d\|\\."
syn region  chestnutStringD	       start=+"+  skip=+\\\\\|\\"+  end=+"\|$+  contains=javaScriptSpecial,@htmlPreproc
syn region  chestnutStringS	       start=+'+  skip=+\\\\\|\\'+  end=+'\|$+  contains=javaScriptSpecial,@htmlPreproc

syn match   chestnutSpecialCharacter "'\\.'"
syn match   chestnutNumber	       "-\=\<\d\+L\=\>\|0[xX][0-9a-fA-F]\+\>"
syn region  chestnutRegexpString     start=+/[^/*]+me=e-1 skip=+\\\\\|\\/+ end=+/[gi]\{0,2\}\s*$+ end=+/[gi]\{0,2\}\s*[;.,)\]}]+me=e-1 contains=@htmlPreproc oneline

syn keyword chestnutConditional	        if else
syn keyword chestnutRepeat		while for
syn keyword chestnutBranch		break continue
"syn keyword chestnutOperator		new delete instanceof typeof
syn keyword chestnutType	       	int real int1d real1d int2d real2d window
syn keyword chestnutStatement		return
syn keyword chestnutBoolean		true false
"syn keyword chestnutNull		null undefined
"syn keyword chestnutIdentifier	arguments this var
"syn keyword chestnutLabel		case default
"syn keyword chestnutException		try catch finally throw
"syn keyword chestnutMessage		alert confirm prompt status
"syn keyword chestnutGlobal		self window top parent
"syn keyword chestnutReserved		abstract boolean byte char class const debugger double enum export extends final float goto implements import int interface long native package private protected public short static super synchronized throws transient volatile 

syn sync fromstart
syn sync maxlines=100

if main_syntax == "chestnut"
  syn sync ccomment chestnutComment
endif

" Define the default highlighting.
" For version 5.7 and earlier: only when not done already
" For version 5.8 and later: only when an item doesn't have highlighting yet
if version >= 508 || !exists("did_chestnut_syn_inits")
  if version < 508
    let did_chestnut_syn_inits = 1
    command -nargs=+ HiLink hi link <args>
  else
    command -nargs=+ HiLink hi def link <args>
  endif
  HiLink chestnutComment		Comment
  HiLink chestnutLineComment		Comment
  HiLink chestnutCommentTodo		Todo
  HiLink chestnutSpecial		Special
  HiLink chestnutStringS		String
  HiLink chestnutStringD		String
  HiLink chestnutCharacter		Character
  HiLink chestnutSpecialCharacter	chestnutSpecial
  HiLink chestnutNumber	        	chestnutValue
  HiLink chestnutConditional		Conditional
  HiLink chestnutRepeat	        	Repeat
  HiLink chestnutBranch	        	Conditional
  HiLink chestnutOperator		Operator
  HiLink chestnutType			Type
  HiLink chestnutStatement		Statement
  HiLink chestnutFunction		Function
  HiLink chestnutBraces	        	Function
  HiLink chestnutError	        	Error
  HiLink chestnutParenError		chestnutError
  HiLink chestnutNull			Keyword
  HiLink chestnutBoolean		Boolean
  HiLink chestnutRegexpString		String

  HiLink chestnutIdentifier		Identifier
  HiLink chestnutLabel	        	Label
  HiLink chestnutException		Exception
  HiLink chestnutMessage		Keyword
  HiLink chestnutGlobal		        Keyword
  HiLink chestnutMember	        	Keyword
  HiLink chestnutDeprecated		Exception 
  HiLink chestnutReserved		Keyword
  HiLink chestnutDebug	        	Debug
  HiLink chestnutConstant		Label

  delcommand HiLink
endif

let b:current_syntax = "chestnut"
if main_syntax == 'chestnut'
  unlet main_syntax
endif

" vim: ts=8

