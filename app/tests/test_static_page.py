import re

from hwsw_app.main import STATIC_DIR


def test_page_is_served_at_the_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "<title>HW/SW Partitioning</title>" in response.text


def test_every_script_and_stylesheet_the_page_loads_exists(client):
    page = client.get("/").text
    references = re.findall(r'(?:src|href)="([^"]+)"', page)
    assert references  # the page does load things
    for reference in references:
        assert client.get("/" + reference).status_code == 200, reference


def test_every_module_the_scripts_import_is_served(client):
    for script in (STATIC_DIR / "js").glob("*.js"):
        for imported in re.findall(r'from "\./([^"]+)"', script.read_text(encoding="utf-8")):
            assert (STATIC_DIR / "js" / imported).is_file(), f"{script.name} imports {imported}"


def test_every_element_the_scripts_look_up_exists_in_the_page(client):
    page = client.get("/").text
    main_js = (STATIC_DIR / "js" / "main.js").read_text(encoding="utf-8")
    looked_up = set(re.findall(r'\$\("([^"]+)"\)', main_js))
    missing = {element_id for element_id in looked_up if f'id="{element_id}"' not in page}
    assert not missing, f"main.js uses ids that index.html lacks: {sorted(missing)}"
