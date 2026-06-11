#!/usr/bin/env python

import pathlib
import textwrap

import git
import packaging.version
from tlz.itertoolz import last


def dev_version(most_recent_release):
    v = packaging.version.parse(most_recent_release)

    next_version = (v.major, v.minor, v.micro + 1)
    return str(v.__replace__(release=next_version, dev=0))


def tagged_date(tag_reference):
    tag = tag_reference.tag
    if tag is None:
        return tag_reference.commit.authored_date

    return tag.tagged_date


def extract_version(tag):
    return str(tag).lstrip("v")


def main():
    root = pathlib.Path.cwd()
    repo = git.Repo(root)

    tags = [extract_version(tag) for tag in sorted(repo.tags, key=tagged_date)]
    fallback_version = "0.0.0"
    most_recent_release = last(tags) if tags else fallback_version
    version = dev_version(most_recent_release)

    recipe_root = root / "ci/rattler-recipe"

    template_path = recipe_root / "recipe_template.yaml"
    template = template_path.read_text()

    context = textwrap.dedent(
        f"""\
        context:
          version: {version}
        """.rstrip(),
    )
    recipe = "\n".join([context, "", template])
    recipe_path = recipe_root / "recipe.yaml"
    recipe_path.write_text(recipe)


if __name__ == "__main__":
    main()
