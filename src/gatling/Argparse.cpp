//
// Copyright (C) 2019-2022 Pablo Delgado Krämer
//
// This program is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// This program is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program. If not, see <https://www.gnu.org/licenses/>.
//

#include "Argparse.h"

#include <pxr/imaging/hd/renderDelegate.h>

PXR_NAMESPACE_OPEN_SCOPE

constexpr static const char* DEFAULT_AOV = "color";
constexpr static int DEFAULT_IMAGE_WIDTH = 800;
constexpr static int DEFAULT_IMAGE_HEIGHT = 800;
constexpr static const char* DEFAULT_CAMERA_PATH = "";
constexpr static bool DEFAULT_GAMMA_CORRECTION = true;

TF_DEFINE_PRIVATE_TOKENS(
  _AppSettingsTokens,
  ((aov, "aov"))                           \
  ((image_width, "image-width"))           \
  ((image_height, "image-height"))         \
  ((camera_path, "camera-path"))           \
  ((gamma_correction, "gamma-correction")) \
  ((help, "help"))
);

namespace
{
  void _PrintCorrectUsage(const HdRenderSettingDescriptorList& renderSettingDescs)
  {
    fflush(stdout);
    printf("Usage: gatling <scene.usd> <render.png> [options]\n");
    printf("\n");

    // Calculate column sizes.
    size_t keyColumnSize = 0;
    size_t nameColumnSize = 0;
    for (const HdRenderSettingDescriptor& desc : renderSettingDescs)
    {
      const std::string& key = desc.key.GetString();
      const std::string& name = desc.name;
      keyColumnSize = std::max(keyColumnSize, key.length());
      nameColumnSize = std::max(nameColumnSize, name.length());
    }
    keyColumnSize += 2;
    nameColumnSize += 2;

    // Table header.
    printf("%-*s%-*s%s\n", (int) keyColumnSize, "Option", (int) nameColumnSize, "Description", "Default value");

    // Print each setting as one row.
    for (const HdRenderSettingDescriptor& desc : renderSettingDescs)
    {
      const char* keyCStr = desc.key.GetText();
      const char* nameCStr = desc.name.c_str();

      bool isValueEmpty = desc.defaultValue.IsEmpty();
      bool isValueBool = desc.defaultValue.IsHolding<bool>();
      bool isValueFloat = desc.defaultValue.IsHolding<double>() || desc.defaultValue.IsHolding<float>() ||
                          desc.defaultValue.IsHolding<pxr_half::half>();
      bool canCastToInt = desc.defaultValue.CanCast<int>();
      bool canCastToString = desc.defaultValue.CanCast<std::string>();

      if (isValueEmpty)
      {
        printf("%-*s%-*s\n", (int) keyColumnSize, keyCStr, (int) nameColumnSize, nameCStr);
      }
      else if (isValueBool)
      {
        auto defaultValue = desc.defaultValue.UncheckedGet<bool>();
        printf("%-*s%-*s%s\n", (int) keyColumnSize, keyCStr, (int) nameColumnSize, nameCStr, defaultValue ? "true" : "false");
      }
      else if (isValueFloat)
      {
        auto defaultValue = VtValue::Cast<float>(desc.defaultValue).UncheckedGet<float>();
        printf("%-*s%-*s%.5f\n", (int) keyColumnSize, keyCStr, (int) nameColumnSize, nameCStr, defaultValue);
      }
      else if (canCastToInt)
      {
        auto defaultValue = VtValue::Cast<int>(desc.defaultValue).UncheckedGet<int>();
        printf("%-*s%-*s%i\n", (int) keyColumnSize, keyCStr, (int) nameColumnSize, nameCStr, defaultValue);
      }
      else if (canCastToString)
      {
        auto defaultValue = VtValue::Cast<std::string>(desc.defaultValue).UncheckedGet<std::string>();
        printf("%-*s%-*s\"%s\"\n", (int) keyColumnSize, keyCStr, (int) nameColumnSize, nameCStr, defaultValue.c_str());
      }
      else
      {
        TF_FATAL_CODING_ERROR("Value for render setting %s can not be displayed!", keyCStr);
      }
    }
    fflush(stdout);
  }

  bool _ParseInt(int* out, const char* in)
  {
    char* end;
    long l = std::strtol(in, &end, 10);
    if (in == end || l < INT_MIN || l > INT_MAX)
    {
      return false;
    }
    *out = (int) l;
    return true;
  }

  bool _ParseFloat(float* out, const char* in)
  {
    char* end;
    *out = std::strtof(in, &end);
    return in != end;
  }

  bool _ParseBool(bool* out, const char* in)
  {
    if (std::strcmp(in, "true") == 0)
    {
      *out = true;
      return true;
    }
    if (std::strcmp(in, "false") == 0)
    {
      *out = false;
      return true;
    }
    return false;
  }
}

bool ParseArgs(int argc, const char* argv[], HdRenderDelegate& renderDelegate, AppSettings& settings)
{
  // Add non-delegate specific options to temporary settings list.
  HdRenderSettingDescriptorList renderSettingDescs = renderDelegate.GetRenderSettingDescriptors();
  renderSettingDescs.push_back(HdRenderSettingDescriptor{"AOV", _AppSettingsTokens->aov, VtValue(DEFAULT_AOV)});
  renderSettingDescs.push_back(HdRenderSettingDescriptor{"Output image width", _AppSettingsTokens->image_width, VtValue(DEFAULT_IMAGE_WIDTH)});
  renderSettingDescs.push_back(HdRenderSettingDescriptor{"Output image height", _AppSettingsTokens->image_height, VtValue(DEFAULT_IMAGE_HEIGHT)});
  renderSettingDescs.push_back(HdRenderSettingDescriptor{"Camera path", _AppSettingsTokens->camera_path, VtValue(DEFAULT_CAMERA_PATH)});
  renderSettingDescs.push_back(HdRenderSettingDescriptor{"Gamma correction", _AppSettingsTokens->gamma_correction, VtValue(DEFAULT_GAMMA_CORRECTION)});
  renderSettingDescs.push_back(HdRenderSettingDescriptor{"Display usage", _AppSettingsTokens->help, VtValue()});

  // We always want to display the options in the same (sorted) order.
  std::sort(renderSettingDescs.begin(), renderSettingDescs.end(), [](const HdRenderSettingDescriptor& descA, const HdRenderSettingDescriptor& descB)
    {
      const std::string& keyAStr = descA.key.GetString();
      const std::string& keyBStr = descB.key.GetString();
      return keyAStr.compare(keyBStr) < 0;
    }
  );

  if (argc < 3)
  {
    _PrintCorrectUsage(renderSettingDescs);
    return false;
  }

  settings.sceneFilePath = std::string(argv[1]);
  settings.outputFilePath = std::string(argv[2]);
  settings.aov = DEFAULT_AOV;
  settings.imageWidth = DEFAULT_IMAGE_WIDTH;
  settings.imageHeight = DEFAULT_IMAGE_HEIGHT;
  settings.cameraPath = DEFAULT_CAMERA_PATH;
  settings.gammaCorrection = DEFAULT_GAMMA_CORRECTION;
  settings.help = false;

  for (int i = 3; i < argc; i++)
  {
    const char* arg = argv[i];

    // Is "--" missing?
    if (strlen(arg) <= 2)
    {
      _PrintCorrectUsage(renderSettingDescs);
      return false;
    }

    arg += 2;

    // Handle application settings.
    if (arg == _AppSettingsTokens->help)
    {
      _PrintCorrectUsage(renderSettingDescs);
      settings.help = true;
      return true;
    }
    else if (arg == _AppSettingsTokens->aov)
    {
      if (i + 1 >= argc)
      {
        _PrintCorrectUsage(renderSettingDescs);
        return false;
      }
      settings.aov = std::string(argv[++i]);
    }
    else if (arg == _AppSettingsTokens->image_width)
    {
      if (i + 1 >= argc || !_ParseInt(&settings.imageWidth, argv[++i]))
      {
        _PrintCorrectUsage(renderSettingDescs);
        return false;
      }
    }
    else if (arg == _AppSettingsTokens->image_height)
    {
      if (i + 1 >= argc || !_ParseInt(&settings.imageHeight, argv[++i]))
      {
        _PrintCorrectUsage(renderSettingDescs);
        return false;
      }
    }
    else if (arg == _AppSettingsTokens->camera_path)
    {
      if (i + 1 >= argc)
      {
        _PrintCorrectUsage(renderSettingDescs);
        return false;
      }
      settings.cameraPath = std::string(argv[++i]);
    }
    else if (arg == _AppSettingsTokens->gamma_correction)
    {
      if (i + 1 >= argc || !_ParseBool(&settings.gammaCorrection, argv[++i]))
      {
        _PrintCorrectUsage(renderSettingDescs);
        return false;
      }
    }
    // Handle delegate settings.
    else
    {
      TfToken settingKey(arg);
      VtValue settingValue = renderDelegate.GetRenderSetting(settingKey);

      // If there is no default value, the setting does not exist.
      if (settingValue.IsEmpty())
      {
        _PrintCorrectUsage(renderSettingDescs);
        return false;
      }

      // If not, we require a value.
      if (i + 1 >= argc)
      {
        _PrintCorrectUsage(renderSettingDescs);
        return false;
      }

      const char* cStr = argv[++i];

#define PARSE_VT_VALUE(TYPE, PARSE_TYPE, PARSE_FN)                 \
      if (settingValue.IsHolding<TYPE>())                          \
      {                                                            \
        PARSE_TYPE t;                                              \
        bool resultOk = PARSE_FN(&t, cStr);                        \
        if (!resultOk)                                             \
        {                                                          \
          _PrintCorrectUsage(renderSettingDescs);                  \
          return false;                                            \
        }                                                          \
        VtValue newValue((TYPE) t);                                \
        renderDelegate.SetRenderSetting(settingKey, newValue);     \
        continue;                                                  \
      }

      PARSE_VT_VALUE(bool,               bool,  _ParseBool)
      PARSE_VT_VALUE(double,             float, _ParseFloat)
      PARSE_VT_VALUE(float,              float, _ParseFloat)
      PARSE_VT_VALUE(pxr_half::half,     float, _ParseFloat)
      PARSE_VT_VALUE(int,                int,   _ParseInt)
      PARSE_VT_VALUE(long,               int,   _ParseInt)
      PARSE_VT_VALUE(unsigned long,      int,   _ParseInt)
      PARSE_VT_VALUE(long long,          int,   _ParseInt)
      PARSE_VT_VALUE(unsigned long long, int,   _ParseInt)
      PARSE_VT_VALUE(int32_t,            int,   _ParseInt)
      PARSE_VT_VALUE(int64_t,            int,   _ParseInt)
      PARSE_VT_VALUE(uint32_t,           int,   _ParseInt)
      PARSE_VT_VALUE(uint64_t,           int,   _ParseInt)

      if (settingValue.IsHolding<std::string>())
      {
        renderDelegate.SetRenderSetting(settingKey, VtValue{std::string(cStr)});
        continue;
      }
      if (settingValue.IsHolding<SdfPath>())
      {
        renderDelegate.SetRenderSetting(settingKey, VtValue{SdfPath(cStr)});
        continue;
      }

      return false;
    }
  }

  return true;
}

PXR_NAMESPACE_CLOSE_SCOPE
