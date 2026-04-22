; Inno Setup script for Omni-ML-GUI
; Build this after generating dist/OmniMLGUI/ via PyInstaller fast mode.

#ifndef AppName
  #define AppName "Omni-ML-GUI"
#endif
#ifndef AppVersion
  #define AppVersion "1.1.0"
#endif
#ifndef AppPublisher
  #define AppPublisher "Taha Ilter Akar"
#endif
#ifndef AppURL
  #define AppURL "https://github.com/tiameister/Omni-ML-GUI"
#endif
#ifndef AppExeName
  #define AppExeName "OmniMLGUI.exe"
#endif
#ifndef AppCopyright
  #define AppCopyright "Copyright (c) 2026 Taha Ilter Akar"
#endif

[Setup]
AppId={{6B67F932-E303-4D66-9D61-B0D60BF7A573}
AppName={#AppName}
AppVersion={#AppVersion}
AppVerName={#AppName} {#AppVersion}
AppPublisher={#AppPublisher}
AppCopyright={#AppCopyright}
AppPublisherURL={#AppURL}
AppSupportURL={#AppURL}
AppUpdatesURL={#AppURL}
DefaultDirName={autopf}\{#AppName}
DefaultGroupName={#AppName}
DisableProgramGroupPage=yes
LicenseFile=LICENSE
OutputDir=dist
OutputBaseFilename={#StringChange(AppExeName, ".exe", "")}-Setup
Compression=lzma2/max
SolidCompression=yes
WizardStyle=modern
ArchitecturesInstallIn64BitMode=x64compatible
PrivilegesRequired=lowest
PrivilegesRequiredOverridesAllowed=dialog
UninstallDisplayIcon={app}\{#AppExeName}
#if FileExists("assets\app.ico")
SetupIconFile=assets\app.ico
#endif

[Languages]
Name: "english"; MessagesFile: "compiler:Default.isl"

[Tasks]
Name: "desktopicon"; Description: "{cm:CreateDesktopIcon}"; GroupDescription: "{cm:AdditionalIcons}"; Flags: unchecked

[Files]
Source: "dist\{#StringChange(AppExeName, ".exe", "")}\*"; DestDir: "{app}"; Flags: ignoreversion recursesubdirs createallsubdirs

[Icons]
Name: "{autoprograms}\{#AppName}"; Filename: "{app}\{#AppExeName}"
Name: "{autodesktop}\{#AppName}"; Filename: "{app}\{#AppExeName}"; Tasks: desktopicon

[Run]
Filename: "{app}\{#AppExeName}"; Description: "{cm:LaunchProgram,{#StringChange(AppName, '&', '&&')}}"; Flags: nowait postinstall skipifsilent
