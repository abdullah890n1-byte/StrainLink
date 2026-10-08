# Building the StrainLink installers

Each installer carries its own Python and libraries: the person installing
StrainLink needs nothing else. An installer must be built on the system it
is for (a Mac cannot build the Windows one, and the reverse).

Every build ends by running the built program's own self-test
(`StrainLink --self-test`: it renders a small synthetic test, analyses it,
builds the report and checks the answers). A build whose self-test fails
must not be given to anyone.

## macOS

On a Mac with this project set up (`.venv`):

    .venv/bin/python -m pip install pyinstaller==6.22.3      (once)
    .venv/bin/python packaging/build_mac.py

Result, in `dist/`:
- `StrainLink.app` (about 280 MB);
- `StrainLink-<version>-macOS-<arm64 or x86_64>.dmg` (about 130 MB): the
  file to give to users. They open it and drag StrainLink onto Applications.

It runs on Macs with the same kind of processor as the one it was built on
(Apple Silicon → `arm64`; an Intel Mac must build its own).

**What a user sees today:** the app is not signed with an Apple Developer ID,
so macOS says it "cannot be opened because the developer cannot be verified".
They must right-click the app, choose Open, then Open again (once). To remove
that warning for customers: join the Apple Developer Program (a yearly fee),
sign the app with the Developer ID certificate, and notarise it with Apple.
`build_mac.py` signs ad hoc; the signing step is the one place to change.

## Windows

On a Windows 10 or 11 PC (64-bit):

1. Install **Python 3.12** from https://www.python.org/downloads/windows/
   (tick "Add python.exe to PATH").
2. Install **Inno Setup 6** (free) from https://jrsoftware.org/isdl.php
   (it turns the program folder into one Setup.exe).
3. Copy the whole StrainLink project folder to the PC.
4. Double-click `packaging\build_windows.bat` and wait (10–20 minutes the
   first time; it downloads the libraries, about 600 MB).

Result: `dist\StrainLink-<version>-Windows-Setup.exe`. Running it installs
StrainLink for the current user (no administrator password), with a Start
menu entry, an optional desktop shortcut and an uninstaller.

Without Inno Setup the script still builds the program in `dist\StrainLink\`
(run `StrainLink.exe`); only the single-file installer is skipped.

**What a user sees today:** Windows SmartScreen warns about an "unknown
publisher" (More info → Run anyway). To remove that: buy a code-signing
certificate and sign `StrainLink.exe` and the Setup.exe.

StrainLink's window uses Microsoft Edge WebView2, which is part of
Windows 11 and of up-to-date Windows 10. On an old Windows 10 without it,
install the "WebView2 Runtime" from Microsoft.

**Not yet tested:** the Windows build script was written on a Mac and has not
been run. The first run on a Windows PC is the test; if it stops, the text in
the window (and `%APPDATA%\StrainLink\selftest.log`) says where.

## Checking any installed copy

    StrainLink --self-test

- macOS: `/Applications/StrainLink.app/Contents/MacOS/StrainLink --self-test`
- Windows: `"%LOCALAPPDATA%\Programs\StrainLink\StrainLink.exe" --self-test`,
  then read `%APPDATA%\StrainLink\selftest.log`

## Before selling

- The tests' real sand photos (PyPIV, GPL) are **not** in the installers: only
  the `strainlink` package and its libraries are bundled. Keep it that way.
- The bundled libraries are under permissive licences (BSD, MIT, Apache,
  PSF, HPND) as far as checked; a full licence review and a third-party
  notices file are still to do (see Tasks in the docs vault).
