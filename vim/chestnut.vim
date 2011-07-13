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
syn match   chestnutLineComment      "\/\/.*" contains=@Spell,chestnutCommentTodo
syn match   chestnutCommentSkip      "^[ \t]*\*\($\|[ \t]\+\)"
syn region  chestnutComment	       start="/\*"  end="\*/" contains=@Spell,chestnutCommentTodo
syn match   chestnutSpecial	       "\\\d\d\d\|\\."
syn region  chestnutStringD	       start=+"+  skip=+\\\\\|\\"+  end=+"\|$+  contains=chestnutSpecial,@htmlPreproc
syn region  chestnutStringS	       start=+'+  skip=+\\\\\|\\'+  end=+'\|$+  contains=chestnutSpecial,@htmlPreproc

syn match   chestnutSpecialCharacter "'\\.'"
syn match   chestnutNumber	       "-\=\<\d\+L\=\>\|0[xX][0-9a-fA-F]\+\>"

syn keyword chestnutConditional	        if else
syn keyword chestnutOperators           and or not
syn keyword chestnutRepeat		while for foreach in end
syn keyword chestnutBranch		break continue

syn keyword chestnutFunctionDecoration  parallel sequential

syn keyword chestnutInts	       	Int   IntArray1d   IntArray2d   IntArray3d   IntWindow1d   IntWindow2d   IntWindow3d
syn keyword chestnutReals               Real  RealArray1d  RealArray2d  RealArray3d  RealWindow1d  RealWindow2d  RealWindow3d
syn keyword chestnutColors              Color ColorArray1d ColorArray2d ColorArray3d ColorWindow1d ColorWindow2d ColorWindow3d
syn keyword chestnutOtherTypes          Bool Point1d Point2d Point3d Size1d Size2d Size3d

syn keyword chestnutStatement		return
syn keyword chestnutBoolean		yes no true false

syn keyword chestnutParallelBuiltin     reduce sort randomInts randomReals
syn keyword chestnutSequentialBuiltin   print location

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
  HiLink chestnutOperators		Operator
  HiLink chestnutStatement		Statement
  HiLink chestnutParallelBuiltin	Function
  HiLink chestnutSequentialBuiltin	Function
  HiLink chestnutBraces	        	Function
  HiLink chestnutError	        	Error
  HiLink chestnutParenError		chestnutError
  HiLink chestnutNull			Keyword
  HiLink chestnutBoolean		Boolean

  HiLink chestnutFunctionDecoration     Keyword

  HiLink chestnutInts		Type
  HiLink chestnutReals  		Type
  HiLink chestnutColors			Type
  HiLink chestnutSizes			Type
  HiLink chestnutOtherTypes		Type

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

