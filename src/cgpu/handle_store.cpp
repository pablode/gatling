#include "handle_store.hpp"

#include <algorithm>
#include <cassert>

using namespace cgpu;

handle_store::handle_store()
  : mMaxIndex{0} {}

handle_store::~handle_store() {}

std::uint64_t handle_store::create()
{
  assert(mMaxIndex < ~0ul);

  if (mFreeIndices.empty())
  {
    const std::uint32_t index = mMaxIndex++;

    if (index >= mVersions.size()) {
      mVersions.resize(index + 1);
    }
    mVersions[index] = 1;

    return make_handle(1, index);
  }
  else
  {
    const std::uint32_t index = mFreeIndices.back();
    mFreeIndices.pop_back();
    const std::uint32_t version = mVersions[index];
    return make_handle(version, index);
  }
}

bool handle_store::is_valid(const std::uint64_t& handle) const
{
  const std::uint32_t version = static_cast<std::uint32_t>(handle >> 32);
  const std::uint32_t index =   static_cast<std::uint32_t>(handle);
  if (index >= mMaxIndex) {
    return false;
  }
  if (mVersions[index] != version) {
    return false;
  }
  return true;
}

void handle_store::free(const std::uint64_t& handle)
{
  const std::uint32_t index = extract_index(handle);
  mVersions[index]++;
  mFreeIndices.push_back(index);
}

std::uint32_t handle_store::extract_index(const std::uint64_t& handle) const {
  return static_cast<std::uint32_t>(handle);
}

std::uint64_t handle_store::make_handle(
  const std::uint32_t& version,
  const std::uint32_t& index) const
{
  return static_cast<std::uint64_t>(index) |
        (static_cast<std::uint64_t>(version) << 32ul);
}
