# Arabic script in Unicode

As of Unicode 18.0, the Arabic script is contained in the following blocks:

- Arabic (0600–06FF, 256 characters)
- Arabic Supplement (0750–077F, 48 characters)
- Arabic Extended-B (0870–089F, 43 characters)
- Arabic Extended-A (08A0–08FF, 96 characters)
- Arabic Presentation Forms-A (FB50–FDFF, 656 characters)
- Arabic Presentation Forms-B (FE70–FEFF, 141 characters)
- Rumi Numeral Symbols (10E60–10E7F, 31 characters)
- Arabic Extended-C (10EC0–10EFF, 60 characters)
- Indic Siyaq Numbers (1EC70–1ECBF, 68 characters)
- Ottoman Siyaq Numbers (1ED00–1ED4F, 61 characters)
- Arabic Mathematical Alphabetic Symbols (1EE00–1EEFF, 143 characters)

The Arabic block encodes the standard letters and diacritics, but does not encode contextual forms (U+0621–U+0652 being directly based on ISO 8859-6); and also includes the most common diacritics and Arabic-Indic digits. Arabic Supplement encodes letter variants mostly used for writing African (non-Arabic) languages. Arabic Extended-B and Arabic Extended-A encode additional Qur'anic annotations and letter variants used for various non-Arabic languages. Arabic Mathematical Alphabetical Symbols encodes characters used in Arabic mathematical expressions. Indic Siyaq Numbers contains a specialized subset of Arabic script that was used for accounting in India under the Mughal Empire by the 17th century through the middle of the 20th century. Ottoman Siyaq Numbers contains a specialized subset of Arabic script, also known as Siyakat numbers, used for accounting in Ottoman Turkish documents.

Arabic Presentation Forms-A encodes contextual forms and ligatures used by older encodings for Persian, Urdu, Sindhi and Central Asian languages. Arabic Presentation Forms-B encodes spacing forms of Arabic diacritics, and more contextual letter forms. The presentation forms are present only for compatibility with older standards, and are not currently needed for correctly rendering text.

## Contextual forms

Below is a demonstration for the basic alphabet used in Modern Standard Arabic illustrating how Arabic letters are expected to appear in different contexts. Codepoints listed as contextual forms "should not be used in general interchange". Unicode has other methods of encoding the difference if necessary, such as Zero-width joiner.

## Punctuation and ornaments

Only the Arabic question mark ⟨؟⟩ and the Arabic comma ⟨،⟩ are used in regular Arabic script typing and the comma is often substituted for the Latin script comma ⟨,⟩ which is also used as the decimal separator when the Eastern Arabic numerals are used (e.g. ⟨100.6⟩ compared to ⟨١٠٠,٦⟩).

- U+060C ، ARABIC COMMA
- U+060D ؍ ARABIC DATE SEPARATOR
- U+060E ؎ ARABIC POETIC VERSE SIGN
- U+060F ؏ ARABIC SIGN MISRA
- U+061B ؛ ARABIC SEMICOLON
- U+061E ؞ ARABIC TRIPLE DOT PUNCTUATION MARK
- U+061F ؟ ARABIC QUESTION MARK
- U+066D ٭ ARABIC FIVE POINTED STAR
- U+06D4 ۔ ARABIC FULL STOP
- U+06DD ۝ ARABIC END OF AYAH
- U+06DE ۞ ARABIC START OF RUB EL HIZB
- U+06E9 ۩ ARABIC PLACE OF SAJDAH
- U+06FD ۽ ARABIC SIGN SINDHI AMPERSAND
- U+FD3E ﴾ Arabic ornate left parenthesis
- U+FD3F ﴿ Arabic ornate right parenthesis

## Word ligatures

Arabic Presentation Forms-A has a few characters defined as "word ligatures" for terms frequently used in formulaic expressions in Arabic. They are rarely used out of professional liturgical typing. Note that the Rial grapheme is normally written fully, not by the ligature.

- U+FDF0 ﷰ ARABIC LIGATURE SALLA USED AS KORANIC STOP SIGN ISOLATED FORM (صلى, stylized as صلے)
- U+FDF1 ﷱ ARABIC LIGATURE QALA USED AS KORANIC STOP SIGN ISOLATED FORM (قلى, stylized as قلے)
- U+FDF2 ﷲ ARABIC LIGATURE ALLAH ISOLATED FORM (اللّٰه)
- U+FDF3 ﷳ ARABIC LIGATURE AKBAR ISOLATED FORM (أكبر), as in the phrase الله أكبر allāhu ʼakbar
- U+FDF4 ﷴ ARABIC LIGATURE MOHAMMAD ISOLATED FORM (مُحَمَّد)
- U+FDF5 ﷵ ARABIC LIGATURE SALAM ISOLATED FORM (صلعم, the abbreviation for صلى الله عليه وسلم 'peace be upon him')
- U+FDF6 ﷶ ARABIC LIGATURE RASOUL ISOLATED FORM (رسول)
- U+FDF7 ﷷ ARABIC LIGATURE ALAYHE ISOLATED FORM (عليه)
- U+FDF8 ﷸ ARABIC LIGATURE WASALLAM ISOLATED FORM (وسلم)
- U+FDF9 ﷹ ARABIC LIGATURE SALLA ISOLATED FORM (صلى)
- U+FDFA ﷺ ARABIC LIGATURE SALLALLAHOU ALAYHE WASALLAM (صَلَّى ٱللَّٰهُ عَلَيْهِ وَسَلَّمَ 'peace be upon him' ṣallā llāhu ʻalayhi wa-sallama)
- U+FDFB ﷻ ARABIC LIGATURE JALLAJALALOUHOU (جَلَّ جَلَالُهُ jalla jalāluhu)
- U+FDFC ﷼ RIAL SIGN (ريال)
- U+FDFD ﷽ ARABIC LIGATURE BISMILLAH AR-RAHMAN AR-RAHEEM (بِسْمِ ٱللَّٰهِ ٱلرَّحْمَٰنِ ٱلرَّحِيمِ bi-smi llāhi r-raḥmāni r-raḥīmi)

## Code blocks

### Arabic Presentation Forms-A

They are mostly ligatures which can be created from the previous charts' characters, with the exception of the bracket-like graphemes ﴾ ﴿ and some of them are ligatures of common liturgical phrases.

### Arabic Presentation Forms-B

These can all be created from the basic chart's characters.
