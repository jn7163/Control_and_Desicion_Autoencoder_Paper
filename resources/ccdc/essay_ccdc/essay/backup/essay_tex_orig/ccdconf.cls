%& -no-cctspace
% LaTeX class for Chinese Control and Decision Conference papers
%

\newif\ifBalanceColumns
\BalanceColumnsfalse % change to true to use 'multicol' package.

%
% Note: when using the 'multicol' package, the 'figure' and 'table'
% environments are modified to ignore user provided placement specifiers
% and always use the '[H]' specifier of the 'float' package.
%

\ProvidesClass{cccconf}[2006/04/15, by Lin-bo Zhang (zlb@lsec.cc.ac.cn)]
\NeedsTeXFormat{LaTeX2e}[1994/12/01]
\DeclareOption*{\PassOptionsToClass{\CurrentOption}{article}}

\newif\ifChinese
\DeclareOption{english}{\Chinesefalse}
\DeclareOption{chinese}{\Chinesetrue}
\ProcessOptions

\ifBalanceColumns
  \LoadClass[a4paper]{article}[1996/10/31]
  \RequirePackage{multicol}
  \RequirePackage{float}
  \AtEndDocument{\end{multicols}}
  % Force 'figure' and 'table' to use [H] placement
  \let\figure@bak\figure
  \renewcommand\figure[1][]{\figure@bak[H]}
  \let\table@bak\table
  \renewcommand\table[1][]{\table@bak[H]}
  \let\balance\relax
\else
  \LoadClass[a4paper,twocolumn]{article}[1996/10/31]
  \RequirePackage{balance}
\fi
\pagestyle{empty}
\RequirePackage{times}

\def\SetEnglishNames{%
  \def\emailname{E-mail}%
  \def\abstractname{Abstract}%
  \def\keywordsname{Key Words}
  \def\arefsuffix{@eng}}

\newif\ifsecondtitle
\ifChinese
  \RequirePackage[CJK]{cctbase}[2003/01/10]
  \RequirePackage[CJKbookmarks]{hyperref}
  \def\emailname{E-mail}
  \def\abstractname{摘^^a1^^a1要}
  \def\keywordsname{关键词}
  \def\tablename{表}
  \def\figurename{图}
  \def\refname{参考文献}
  \def\arefsuffix{@chn}
  \let\biaosong\heiti
  \secondtitletrue
\else
%  \RequirePackage{hyperref}
  \SetEnglishNames
  \secondtitlefalse
\fi

\renewcommand\thefootnote{}
%\newcommand\URL[1]{\href{#1}{{#1}}}
\newcommand\URL[1]{#1}

%---------------------------------- Page layout
\RequirePackage{graphicx}
\RequirePackage[left=1.9cm,right=1.9cm,top=2cm,bottom=2.7cm]{geometry}

\ifChinese\renewcommand\baselinestretch{1.1}\fi
\columnsep=0.7cm
%\parskip=6\p@

\tolerance=5000
\hbadness=5000
\vbadness=5000

\ifChinese
  %\let\small@bak\small
  %\def\small{\small@bak\zihao{6}}

  %\let\normalsize@bak\normalsize
  %\def\normalsize{\normalsize@bak\zihao{-5}}

  \let\footnotesize@bak\footnotesize
  \def\footnotesize{\footnotesize@bak\zihao{6}}
  \AtBeginDocument{\parindent=2\ccwd}
\else
  \AtBeginDocument{\parindent=\z@}
\fi

%---------------------------------- Title, abstract, and key words
% (Note: \keywords must be present)

\newbox\titleb@x
\setbox\titleb@x\vbox{}

\newcounter{affno}
\setcounter{affno}{0}

\def\aref#1{\let\@aref@comma\relax{\small$^{\@aref#1,,}$}}

\def\@aref#1,{\def\@tempa{#1}\ifx\@tempa\@empty \let\@next\relax\else
  \@aref@comma\ref{#1\arefsuffix}\let\@next\@aref\fi\let\@aref@comma,\@next}

\newtoks\@afflist
\@afflist={}
\newcommand\affiliation[2][]{\@afflist=\expandafter{\the\@afflist \par
  \fontsize{9}{10}\selectfont \def\@tempa{#1}\ifx\@tempa\@empty\else
    \refstepcounter{affno}\label{#1\arefsuffix}\arabic{affno}. \fi
  #2\par\smallskip}}

\def\maketitle{\def\saved@title{\let \footnote \thanks \parskip\z@
  \begin{center}
    {\Large\bfseries
     \ifChinese
      \vspace*{1.2cm}%
      \heiti \zihaoAny{18}%
      \ifsecondtitle \else
        \fontsize{14}{16}\selectfont %16,20
      \fi
     \else
      \vspace*{1cm}%\vspace*{-0.2cm}
      \fontsize{16}{20}\selectfont   
     \fi
     \@title\par}\bigskip
    {\fontsize{11}{12}\selectfont\ifChinese\large\fangsong\fi \@author}\\ %normalsize=>11pt
    \smallskip
    \the\@afflist 
  \end{center}
  \par\bigskip}}

\newbox\@abstract
\renewenvironment{abstract}{\global\setbox\@abstract\vbox\bgroup
  \hsize=\textwidth\leftskip=1cm \rightskip=1cm \noindent
  \small \textbf{\abstractname: }}{\egroup}

\def\keywords#1{
  \setbox\titleb@x\vbox{\hsize=\textwidth \box\titleb@x
  \def\email##1{\unskip\\\emailname: {##1}}%
  \saved@title\box\@abstract\par\medskip
  \vbox{\leftskip=1cm \rightskip=1cm \small
    \noindent\textbf{\keywordsname: }#1}\par
  \ifsecondtitle \vspace*{-1.5cm}%\vspace*{-0.5cm}
  \fi
  \bigskip}%
  \@afflist={}\setcounter{affno}{0}%
  \ifsecondtitle
    \SetEnglishNames
    \secondtitlefalse
  \else
    \ifBalanceColumns
%        \begin{flushleft}
%          \footnotesize
%        \end{flushleft}
%        \par\vskip 0cm %
        \box\titleb@x\par\vskip 0.8cm
      \begin{multicols}{2}
      \let\@footnotetext\orig@footnotetext
    \else
      \twocolumn[%
%        \begin{flushleft}
%          \footnotesize
%        \end{flushleft}
%        \par\vskip 0cm %
        \box\titleb@x\par\vskip 0.8cm
      ]%
    \fi
    \setbox\titleb@x\vbox{}%
  \fi
}

%---------------------------------- Sections

\renewcommand\section{\@startsection{section}{1}{\z@}%
                                    {-6pt \@plus -1ex \@minus -1ex}%
                                    {6pt \@plus .2ex}%
                                    {\normalfont\fontsize{11}{12}\bfseries}}
\renewcommand\subsection{\@startsection{subsection}{2}{\z@}%
                                       {-6pt }% \@plus -1ex \@minus -1ex}%
                                       {3pt \@plus .0ex}%
                                       {\normalfont\normalsize\bfseries}}
\ifChinese
  % Restore section definition
  \def\@seccntformat#1{\csname the#1\endcsname\space\space}
  \def\@sect#1#2#3#4#5#6[#7]#8{%
    \ifnum #2>\c@secnumdepth
      \let\@svsec\@empty
    \else
      \refstepcounter{#1}%
      \protected@edef\@svsec{\@seccntformat{#1}\relax}%
    \fi
    \@tempskipa #5\relax
    \ifdim \@tempskipa>\z@
      \begingroup
        #6{%
          \@hangfrom{\hskip #3\relax\@svsec}%
            \interlinepenalty \@M #8\@@par}%
      \endgroup
      \csname #1mark\endcsname{#7}%
      \addcontentsline{toc}{#1}{%
        \ifnum #2>\c@secnumdepth \else
          \protect\numberline{\csname the#1\endcsname}%
        \fi
        #7}%
    \else
      \def\@svsechd{%
        #6{\hskip #3\relax
        \@svsec #8}%
        \csname #1mark\endcsname{#7}%
        \addcontentsline{toc}{#1}{%
          \ifnum #2>\c@secnumdepth \else
            \protect\numberline{\csname the#1\endcsname}%
          \fi
          #7}}%
    \fi
    \@xsect{#5}}
  %\renewcommand\thesection{\arabic{section}}
  %\renewcommand\thesubsection{\thesection.\arabic{subsection}}
  %\renewcommand\thesubsubsection{\thesubsection.\arabic{subsubsection}}
\else
  \renewcommand\refname{REFERENCES}
\fi

%---------------------------------- Top/bottom rules in tabulars
%%%\newcommand\hhline{%
%%%  \noalign{\ifnum0=`}\fi\hrule \@height 2\arrayrulewidth \futurelet
%%%   \@tempa\@xhline}
\newcommand\hhline{\hline\noalign{\vskip -2pt}\hline\noalign{\vskip 1pt}}

%---------------------------------- Floats
% Note: the values below are taken from ieeeconf.cls

\floatsep 1\baselineskip plus  0.2\baselineskip minus  0.2\baselineskip
\textfloatsep 1.7\baselineskip plus  0.2\baselineskip minus  0.4\baselineskip
\@fptop 0pt plus 1fil
\@fpsep 0.75\baselineskip plus 2fil
\@fpbot 0pt plus 1fil
\def\topfraction{1.0}
\def\bottomfraction{.4}
\def\floatpagefraction{0.8}
\def\textfraction{.2}

\dblfloatsep 1\baselineskip plus  0.2\baselineskip minus  0.2\baselineskip
\dbltextfloatsep 1.7\baselineskip plus  0.2\baselineskip minus  0.4\baselineskip

\@dblfptop 0pt plus 1fil
\@dblfpsep 0.75\baselineskip plus 2fil
\@dblfpbot 0pt plus 1fil
\def\dbltopfraction{1.0}
\def\dblfloatpagefraction{0.8}
\setcounter{dbltopnumber}{4}

\intextsep 1\baselineskip plus 0.2\baselineskip minus  0.2\baselineskip
\setcounter{topnumber}{2}
\setcounter{bottomnumber}{2}
\setcounter{totalnumber}{4}

\setlength\abovecaptionskip{0.25\baselineskip}
\setlength\belowcaptionskip{0.25\baselineskip}

%---------------------------------- The bibliography

\renewcommand\@openbib@code{\parsep \z@ \itemsep \z@ \parskip \z@
  \small\ifChinese\zihao{-5}\fi}
\renewcommand\newblock{\par}

%---------------------------------- Floats

\def \@floatboxreset {%
  \reset@font
  \small
  \ifChinese\renewcommand\arraystretch{1.2}\fi
  \@setminipage
}

\endinput
