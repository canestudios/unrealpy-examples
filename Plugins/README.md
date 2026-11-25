# Plugin setup

Install UnrealPy for Unreal Engine 5.8 using the [installation guide](https://unrealpy.canestudios.xyz/docs/getting-started/installation/). The Fab package loads its required ScriptCore modules from the UnrealPy plugin. The example project enables UnrealPy in its project descriptor.

For a development checkout with separate plugin descriptors, install both `Plugins/UnrealPy` and `Plugins/ScriptCore` from the same build. Both folders are git-ignored. Match the engine version used to compile the binaries.
