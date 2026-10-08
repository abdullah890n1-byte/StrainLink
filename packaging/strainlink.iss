; Inno Setup recipe for the StrainLink installer (built by packaging\build_windows.bat).
; Installs for the current user only, so no administrator password is needed on lab computers.

#ifndef AppVersion
  #define AppVersion "0.1.0"
#endif

[Setup]
AppId={{8C1B6B7E-5E0B-4F5C-9C39-2B7B7A3E5D41}
AppName=StrainLink
AppVersion={#AppVersion}
AppPublisher=StrainLink
DefaultDirName={autopf}\StrainLink
DefaultGroupName=StrainLink
DisableProgramGroupPage=yes
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
ArchitecturesAllowed=x64compatible
ArchitecturesInstallIn64BitMode=x64compatible
OutputDir=..\dist
OutputBaseFilename=StrainLink-{#AppVersion}-Windows-Setup
SetupIconFile=..\build\icons\StrainLink.ico
UninstallDisplayIcon={app}\StrainLink.exe
Compression=lzma2
SolidCompression=yes
WizardStyle=modern

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "Create a shortcut on the desktop"; GroupDescription: "Shortcuts:"; Flags: unchecked

[Files]
Source: "..\dist\StrainLink\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{group}\StrainLink"; Filename: "{app}\StrainLink.exe"
Name: "{group}\Uninstall StrainLink"; Filename: "{uninstallexe}"
Name: "{autodesktop}\StrainLink"; Filename: "{app}\StrainLink.exe"; Tasks: desktopicon

[Run]
Filename: "{app}\StrainLink.exe"; Description: "Open StrainLink now"; Flags: nowait postinstall skipifsilent
