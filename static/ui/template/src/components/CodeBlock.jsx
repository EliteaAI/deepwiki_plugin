import { useMemo } from 'react';
import CodeMirror from '@uiw/react-codemirror';
import { javascript } from '@codemirror/lang-javascript';
import { python } from '@codemirror/lang-python';
import { html } from '@codemirror/lang-html';
import { css } from '@codemirror/lang-css';
import { json } from '@codemirror/lang-json';
import { oneDark } from '@codemirror/theme-one-dark';
import { EditorView } from '@codemirror/view';
import { Box } from '@mui/material';

const languageExtensions = {
  javascript: javascript(),
  js: javascript(),
  jsx: javascript({ jsx: true }),
  typescript: javascript({ typescript: true }),
  ts: javascript({ typescript: true }),
  tsx: javascript({ typescript: true, jsx: true }),
  python: python(),
  py: python(),
  html: html(),
  css: css(),
  json: json(),
};

const CodeBlock = ({ code, language, mode }) => {
  const extension = useMemo(() => {
    return languageExtensions[language] || [];
  }, [language]);

  const theme = mode === 'dark' ? oneDark : undefined;

  return (
    <CodeMirror
      value={code}
      extensions={[extension, EditorView.lineWrapping]}
      theme={theme}
      editable={false}
      basicSetup={{
        lineNumbers: true,
        highlightActiveLineGutter: false,
        highlightActiveLine: false,
        foldGutter: false,
        dropCursor: false,
        allowMultipleSelections: false,
        indentOnInput: false,
        bracketMatching: true,
        closeBrackets: false,
        autocompletion: false,
        rectangularSelection: false,
        crosshairCursor: false,
        highlightSelectionMatches: false,
        closeBracketsKeymap: false,
        searchKeymap: false,
        foldKeymap: false,
        completionKeymap: false,
        lintKeymap: false,
      }}
      style={{
        fontSize: '14px',
        fontFamily: '"Fira Code", "Courier New", monospace',
        borderRadius: '4px',
        border: mode === 'dark' ? '1px solid #3B3E46' : '1px solid #E1E5E9',
        marginTop: '16px',
        marginBottom: '16px',
      }}
    />
  );
};

export default CodeBlock;
