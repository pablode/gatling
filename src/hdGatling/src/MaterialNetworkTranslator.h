#pragma once

#include <pxr/usd/sdf/path.h>

#include <MaterialXCore/Document.h>

#include <memory>

struct gi_material;

PXR_NAMESPACE_OPEN_SCOPE

struct HdMaterialNetwork2;

class MaterialNetworkTranslator
{
public:
  MaterialNetworkTranslator(const std::string& mtlxLibPath);

  gi_material* ParseNetwork(const SdfPath& id,
                            const HdMaterialNetwork2& network) const;

private:
  gi_material* TryParseMdlNetwork(const HdMaterialNetwork2& network) const;

  gi_material* TryParseMtlxNetwork(const SdfPath& id, const HdMaterialNetwork2& network) const;

  MaterialX::DocumentPtr CreateMaterialXDocumentFromNetwork(const SdfPath& id,
                                                            const HdMaterialNetwork2& network) const;

private:
  MaterialX::DocumentPtr m_nodeLib;
};

PXR_NAMESPACE_CLOSE_SCOPE
