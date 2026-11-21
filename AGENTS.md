# AGENTS.md

## Build

gatling uses CMake as a build system generator, and C++20 as the programming language.
The preferred build system of choice is Ninja and the preferred build configuration is Develop.
Besides git submodules in the 'extern' folder, it requires OpenUSD and MDL as external dependencies.
gatling is primarily a Hydra render delegate, thus it needs to be installed into USD's directory.

```sh
git submodule update --init --recursive # once, after clone
cmake -GNinja -S . -B BUILD -DUSD_ROOT=$USD_ROOT -DMDL_ROOT=$MDL_ROOT -DCMAKE_INSTALL_PREFIX=$USD_ROOT/plugin/usd
cmake --build BUILD -j 8 --config Develop
cmake --install BUILD --component hdGatling
```

## Architecture

The code base is horizontal and component-based. Vertical dependency chains are avoided.
Public class names are prefixed by the component name, typically two to four characters. Corresponding files don't have a prefix.

## C++ Code Style

- C++, but we don't want to write modern C++. It should be more like C with classes.
- Always use spaces, no tabs.
- Always add braces if they are optional. Especially for if-statements.
- Class names in PascalCase, variables in camelCase.
- Global constants should be written in UPPER_CASE.
- Only use auto if the type is stated on the right side of the assignment.
- Don't use auto for long names - shorten them using aliases.
- Use comments as rarely as possible. Comments should only add important information if absolutely necessary.
- Don't decorate comments. Use in-line and single-line comments.
- Static functions are placed in an anonymous namespaces outside the main namespace and follow `_PascalCase` naming.
- A namespace indents its content. There is no empty line after declaration and before close.
- Avoid blocks of empty lines.
- Avoid deep identations. Early-abort when possible.
- Match surrounding code and keep changes focused. Do not commit unless asked.
- Function implementations in headers should be avoided.
- Use forward declarations in headers when possible and if the overhead is small.
- Avoid complex logic in constructors and destructors. Use factory functions for initialization routines.
- Only use const keyword for pointers and references. Avoid it for local variables of other data types.
- Curly brackets should always be placed on a new line, except for lambdas.
- Avoid the #ifndef #define pattern for compiler defines.
- Float zeros should not be abbreviated but written out.
- Avoid including headers that are already transitively included.
- When you create a new file, add the license notice with the current year.
- Use std::format over snfprintf.

The following is true for all components except hdGatling:
- File names in PascalCase, functions in camelCase.
- Class members are prefixed `m_`.
- Code should be placed in the `gtl` namespace, unless it's an extension of a third-party library.

hdGatling follows the OpenUSD naming scheme:
- File names are in camelCase, functions in PascalCase.
- Member variables start with an underscore, not an m_.

## GLSL Code Style

- Use camelCase for functions and variable declarations. PascalCase for binding and UPPER_CASE for constants.

## Tests

- Ensure that tests directly or indirectly related to your changes pass.
