from __future__ import annotations

import re
import unicodedata


class Normalizer:
    """
    Text normalizer for tokenizer preprocessing.

    The normalizer applies a small set of common text-cleanup steps such as
    Unicode normalization, quote normalization, line-ending cleanup, and
    trailing whitespace trimming.
    """

    def __init__(
            self,
            unicode_norm: bool = True,
            quote_norm: bool = True,
            line_ending_norm: bool = True,
            whitespace_norm: bool = True
    ):
        """
        Initialize the normalizer configuration.

        Args:
            unicode_norm:
                Whether to normalize Unicode characters.

            quote_norm:
                Whether to normalize curly quotes to standard quotes.

            line_ending_norm:
                Whether to normalize line endings to ``\n``.

            whitespace_norm:
                Whether to trim trailing whitespace from each line.
        """
        self.unicode_norm = unicode_norm
        self.quote_norm = quote_norm
        self.line_ending_norm = line_ending_norm
        self.whitespace_norm = whitespace_norm

        self.CURLY_QUOTES = {
            "‘": "'", "’": "'", "‚": "'", "‛": "'",
            "“": '"', "”": '"', "„": '"', "‟": '"',
            "‹": "'", "›": "'", "«": '"', "»": '"',
        }


    def _unicode(self, text: str) -> str:
        """
        Normalize Unicode text using the NFC form.

        Args:
            text:
                Input string to normalize.

        Returns:
            Normalized string.
        """
        return unicodedata.normalize("NFC", text)


    def _quote(self, text: str) -> str:
        """
        Convert curly quotes to standard ASCII quotes.

        Args:
            text:
                Input string containing smart quotes.

        Returns:
            String with normalized quote characters.
        """

        return re.sub(r"[‘’‚‛“”„‟‹›«»]", lambda m: self.CURLY_QUOTES[m.group()], text)


    def _line_ending(self, text: str) -> str:
        """
        Normalize line endings to a consistent newline format.

        Args:
            text:
                Input string.

        Returns:
            String with normalized line endings.
        """
        return text.replace("\r\n", "\n").replace("\r", "\n")


    def _whitespace(self, text: str) -> str:
        """
        Remove trailing whitespace from each line.

        Args:
            text:
                Input string.

        Returns:
            String with trailing whitespace removed.
        """
        return re.sub(r"[ \t]+$", "", text, flags=re.MULTILINE)


    def normalize(self, text: str) -> str:
        """
        Normalize text according to the configured pipeline.

        Pipeline order:

        1. Unicode
        2. Line endings
        3. Quotes
        4. Whitespace
        """
        if self.unicode_norm:
            text = self._unicode(text=text)

        if self.line_ending_norm:
            text = self._line_ending(text=text)

        if self.quote_norm:
            text = self._quote(text=text)

        if self.whitespace_norm:
            text = self._whitespace(text=text)

        return text